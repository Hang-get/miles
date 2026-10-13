import logging
from argparse import Namespace
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update import checksum_utils
from miles.backends.training_utils.weight_update.protocols.p2p_transfer_utils import RemoteWeightInfo
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater


logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater(_RolloutCellUpdater):
    def __init__(
        self,
        args: Namespace,
        cell_id: str,
        api_client: SGLangApiClient,
        selector: str,
    ) -> None:
        super().__init__(args=args, cell_id=cell_id, api_client=api_client)
        self.selector = selector
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.targets_by_rollout_engine_rank: dict[int, RemoteWeightInfo] = {}
        self._pending_writes: list[Future[None]] = []

    def submit_write(
        self,
        rollout_engine_rank: int,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
        transfer_engine: Any,
        sent_checksums: dict[str, str] | None,
    ) -> None:
        if self.is_errored:
            return
        self._pending_writes.append(
            self._executor.submit(
                self._write_if_active,
                transfer_engine,
                self.targets_by_rollout_engine_rank[rollout_engine_rank],
                names,
                weight_memory_registry,
                rollout_engine_rank=rollout_engine_rank,
                sent_checksums=sent_checksums,
            )
        )

    def wait_for_pending_writes(self, timeout: float) -> None:
        if self.is_errored:
            return
        pending, self._pending_writes = self._pending_writes, []
        for i, future in enumerate(pending):
            try:
                future.result(timeout=timeout)
            except Exception as error:
                self.mark_errored(error)
                for remaining in pending[i + 1 :]:
                    remaining.cancel()
                return

    def _write_if_active(
        self,
        transfer_engine: Any,
        target: RemoteWeightInfo,
        names: list[str],
        weight_memory_registry: dict[str, tuple[int, int, int]],
        rollout_engine_rank: int,
        sent_checksums: dict[str, str] | None,
    ) -> None:
        if self.is_errored:
            logger.warning(f"[P2P-Shared] skipping a queued write to rollout cell {self.cell_id}")
            return
        _do_p2p_write_one_session(
            transfer_engine=transfer_engine,
            remote_session=target,
            names=names,
            weight_memory_registry=weight_memory_registry,
            cell_id=self.cell_id,
        )
        if sent_checksums is not None:
            _verify_transfer_checksums(
                self, rollout_engine_rank=rollout_engine_rank, names=names, sent_checksums=sent_checksums
            )


def _verify_transfer_checksums(
    cell_updater: _P2PRolloutCellUpdater,
    *,
    rollout_engine_rank: int,
    names: list[str],
    sent_checksums: dict[str, str],
) -> None:
    engine_body = cell_updater.submit_client_call(
        "check_weights", action="raw_checksum", names=names, selector=cell_updater.selector
    ).result()
    if engine_body is None:
        return
    checksum_utils.verify_transfer_checksums(
        sent_checksums=sent_checksums, engine_body=engine_body, cell_id=cell_updater.cell_id, rank=rollout_engine_rank
    )
