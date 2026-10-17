import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future

import torch
import zmq
from sglang.srt.weight_sync.colocated_param_bytes import Ack, Closed, End, Hello, Shared, Write

from miles.backends.megatron_utils.sglang import MultiprocessingSerializer, monkey_patch_torch_reductions
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.protocols.transports.remote_shard import (
    RemoteShard,
    RemoteWeightLocation,
    query_remote_shards,
)
from miles.backends.training_utils.weight_update.protocols.transports.rollout_engine_write_threads import (
    RolloutEngineWriteThreads,
)
from miles.backends.training_utils.weight_update.protocols.utils.rollout_engine_rank_assignment import (
    RolloutEngineRankAssignment,
)


class LocalPullTransport:
    """Hands param bytes to the rollout engine rank on this trainer process's own GPU, which pulls them
    (sglang `update_weights_from_colocated_param_bytes`).

    The engine's weights live in torch_memory_saver memory, which another process cannot map, so the engine opens
    this process's transfer buffers over CUDA IPC and copies each written byte range into its published weights.
    Per sync: `begin_session` once the engine's pull op is running, `write` per loaded buffer, and `end_session`
    after every write returned. The socket stays bound for the process's lifetime because the engine keeps its end.
    """

    def __init__(self, *, timeout_s: float) -> None:
        self.transfer_buffer_device = torch.device("cuda", torch.cuda.current_device())
        self.device_uuid = get_device_uuid()
        self._timeout_s = timeout_s
        self._write_threads = RolloutEngineWriteThreads(thread_name_prefix="colocated-pull-write")
        self._transfer_buffers: list[torch.Tensor] = []
        self._sock = zmq.Context.instance().socket(zmq.PAIR)
        self._sock.setsockopt(zmq.LINGER, 0)
        self.socket_path = f"ipc://{tempfile.gettempdir()}/miles-colocated-pull-{uuid.uuid4().hex}.sock"
        self._sock.bind(self.socket_path)

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        assignments: Sequence[RolloutEngineRankAssignment],
        runner_roles: Sequence[str],
    ) -> dict[str, dict[int, list[RemoteShard]]]:
        """Return assigned shards by runner role and rollout rank."""
        self._write_threads.restart()
        return query_remote_shards(rollout_engines, assignments, runner_roles, self._build_remote_shard)

    def register_memory(self, tensor: torch.Tensor) -> None:
        self._transfer_buffers.append(tensor)

    def unregister_memory(self, tensor: torch.Tensor) -> None:
        self._transfer_buffers = [buffer for buffer in self._transfer_buffers if buffer is not tensor]

    def begin_session(
        self, *, expected_names_by_role: Mapping[str, frozenset[str]], stop_waiting: Callable[[], bool]
    ) -> None:
        """Answers the engine's Hello with the transfer buffers and the params its loader writes; `stop_waiting`
        ends the wait early when the engine's pull op already returned."""
        hello = self._recv(stop_waiting=stop_waiting)
        assert isinstance(hello, Hello), f"expected the engine's Hello, got {hello!r}"
        assert hello.device_uuid == self.device_uuid, (
            f"the engine rank on GPU {hello.device_uuid} connected to the trainer rank on GPU {self.device_uuid}; "
            "a pull copies from this GPU's memory, so both must run on the same GPU"
        )
        monkey_patch_torch_reductions()
        self._sock.send_pyobj(
            Shared(
                serialized_buffers=MultiprocessingSerializer.serialize(self._transfer_buffers),
                expected_names_by_role=dict(expected_names_by_role),
            )
        )

    def write(self, remote_shards: Sequence[RemoteShard], tensors_by_name: dict[str, torch.Tensor]) -> list[Future]:
        """Tells the engine to copy each tensor, a view into one transfer buffer, into the weight of the same name;
        returns one future, done once the engine acknowledged the copy."""
        # the write thread sends only after the load that filled the buffer has finished on the GPU
        loaded = torch.cuda.Event()
        loaded.record()
        return self._write_threads.write_each_shard(
            remote_shards,
            tensors_by_name,
            lambda remote_shard, tensors: self._write_shard(remote_shard, tensors, loaded),
        )

    def end_session(self) -> None:
        """Waits until the engine closed its handles to the transfer buffers, so they may be freed."""
        self._sock.send_pyobj(End())
        closed = self._recv(stop_waiting=lambda: False)
        assert isinstance(closed, Closed), f"expected the engine's Closed, got {closed!r}"

    def _write_shard(
        self, remote_shard: RemoteShard, tensors_by_name: dict[str, torch.Tensor], loaded: torch.cuda.Event
    ) -> None:
        (buffer_index,) = {self._find_buffer_index(tensor) for tensor in tensors_by_name.values()}
        buffer_address = self._transfer_buffers[buffer_index].data_ptr()
        loaded.synchronize()
        self._sock.send_pyobj(
            Write(
                buffer_index=buffer_index,
                role=remote_shard.runner_role,
                params=[
                    (name, tensor.data_ptr() - buffer_address, tensor.nbytes)
                    for name, tensor in tensors_by_name.items()
                ],
            )
        )
        ack = self._recv(stop_waiting=lambda: False)
        assert isinstance(ack, Ack), f"expected the engine's Ack, got {ack!r}"
        if ack.error is not None:
            raise RuntimeError(f"the colocated engine rank failed to copy: {ack.error}")

    def _find_buffer_index(self, tensor: torch.Tensor) -> int:
        (buffer_index,) = [
            index
            for index, buffer in enumerate(self._transfer_buffers)
            if buffer.data_ptr() <= tensor.data_ptr() < buffer.data_ptr() + buffer.nbytes
        ]
        return buffer_index

    def _recv(self, *, stop_waiting: Callable[[], bool]):
        deadline = time.monotonic() + self._timeout_s
        while not self._sock.poll(timeout=100):
            if stop_waiting():
                raise RuntimeError("the colocated engine rank's pull op returned before it connected")
            if time.monotonic() > deadline:
                raise TimeoutError(f"no message from the colocated engine rank within {self._timeout_s}s")
        return self._sock.recv_pyobj()

    def _build_remote_shard(
        self, rank_info: dict, rollout_engine_ind: int, rollout_engine_rank: int, runner_role: str
    ) -> RemoteShard:
        return RemoteShard(
            rollout_engine_ind=rollout_engine_ind,
            rollout_engine_rank=rollout_engine_rank,
            runner_role=runner_role,
            transport_peer=self.socket_path,
            weight_locations_by_name={
                name: RemoteWeightLocation(*location) for name, location in rank_info["weight_locations"].items()
            },
        )


def get_device_uuid() -> str:
    # the key sglang's engine ranks look themselves up by (current_platform.get_device_uuid)
    return str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
