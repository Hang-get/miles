from argparse import Namespace
from concurrent.futures import Future

import pytest
import torch
from sglang.srt.utils import MultiprocessingSerializer
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

from miles.backends.training_utils.weight_update.protocols import shm
from miles.backends.training_utils.weight_update.protocols.shm import UpdateWeightFromSharedMemory

_BUCKETS = [
    [
        ("layers.0.w", torch.arange(6, dtype=torch.float32).reshape(2, 3)),
        ("norm.w", torch.ones(3, dtype=torch.bfloat16)),
    ],
    [("layers.1.w", torch.full((4,), 7.0))],
    [("lm_head.w", torch.full((2, 2), -1.0, dtype=torch.bfloat16))],
]


class _RecordingCellUpdater:
    """Maps every TP rank's payload while the call is in flight, as the engine does."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.rebuilt: list[list[list[tuple[str, torch.Tensor]]]] = []

    def submit_client_call(self, name: str, **kwargs) -> Future:
        self.calls.append((name, kwargs))
        self.rebuilt.append([_rebuild(serialized) for serialized in kwargs["serialized_named_tensors"]])
        done: Future = Future()
        done.set_result({"success": True})
        return done


def _rebuild(serialized: str) -> list[tuple[str, torch.Tensor]]:
    payload = MultiprocessingSerializer.deserialize(serialized)
    bucket = FlattenedTensorBucket(flattened_tensor=payload["flattened_tensor"], metadata=payload["metadata"])
    return [(name, tensor.clone()) for name, tensor in bucket.reconstruct_tensors()]


class _FakeDist:
    def barrier(self, group=None) -> None:
        pass


@pytest.fixture(autouse=True)
def _single_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(shm, "dist", _FakeDist())
    monkeypatch.setattr(shm, "get_gloo_group", lambda: None)


def _protocol(*, replica_rank: int, replica_size: int, tp_sizes: list[int]) -> UpdateWeightFromSharedMemory:
    protocol = UpdateWeightFromSharedMemory(Namespace())
    protocol.cell_updaters_of_cell_id = {f"cell-{i}": _RecordingCellUpdater() for i in range(len(tp_sizes))}
    protocol._engine_tp_sizes = tp_sizes
    protocol._selector = "all"
    protocol._replica_rank, protocol._replica_size = replica_rank, replica_size
    return protocol


def _sync(protocol: UpdateWeightFromSharedMemory) -> None:
    protocol.begin_sync(weight_version=1, iter_buckets=None)
    for bucket in _BUCKETS:
        protocol.send_bucket(bucket)
    protocol.after_base_weights()
    protocol.finalize(weight_version=1)


def test_every_engine_rank_rebuilds_the_bucket_tensors():
    """The payload the engine deserializes must reconstruct each tensor with its name, dtype and values."""
    protocol = _protocol(replica_rank=0, replica_size=1, tp_sizes=[1, 2])
    _sync(protocol)

    for updater, tp_size in zip(protocol.cell_updaters_of_cell_id.values(), [1, 2], strict=True):
        assert [name for name, _ in updater.calls] == ["update_weights_from_tensor"] * len(_BUCKETS)
        assert all(kwargs["load_format"] == "flattened_bucket" for _, kwargs in updater.calls)
        for per_rank, bucket in zip(updater.rebuilt, _BUCKETS, strict=True):
            assert len(per_rank) == tp_size
            for rebuilt in per_rank:
                assert [name for name, _ in rebuilt] == [name for name, _ in bucket]
                for (_, got), (_, want) in zip(rebuilt, bucket, strict=True):
                    assert got.dtype == want.dtype
                    torch.testing.assert_close(got, want)


def test_a_replica_rank_publishes_only_its_own_buckets():
    protocol = _protocol(replica_rank=1, replica_size=2, tp_sizes=[1])
    _sync(protocol)

    (updater,) = protocol.cell_updaters_of_cell_id.values()
    assert [[name for name, _ in rebuilt] for (rebuilt,) in updater.rebuilt] == [["layers.1.w"]]
