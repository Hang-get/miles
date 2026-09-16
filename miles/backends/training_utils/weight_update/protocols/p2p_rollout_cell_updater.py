from concurrent.futures import Future
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

from .p2p_transfer_utils import P2PTransferManager, RemoteWeightInfo


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater(_RolloutCellUpdater):
    def __init__(
        self,
        cell_id: str,
        api_client: SGLangApiClient,
    ) -> None:
        super().__init__(cell_id=cell_id, api_client=api_client)
        self.targets_by_rollout_engine_rank: dict[int, RemoteWeightInfo] = {}
        self._pending_writes: list[Future[None]] = []

    def submit_write(
        self,
        rollout_engine_rank: int,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
        transfer_engine: Any,
        transfer_manager: P2PTransferManager,
    ) -> None:
        self._pending_writes.append(
            transfer_manager.submit(
                _do_p2p_write_one_session,
                transfer_engine,
                self.targets_by_rollout_engine_rank[rollout_engine_rank],
                names,
                weight_memory_registry,
            )
        )

    def wait_for_pending_writes(self) -> None:
        pending, self._pending_writes = self._pending_writes, []
        for future in pending:
            future.result()
