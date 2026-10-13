import logging
from argparse import Namespace
from collections.abc import Callable
from concurrent.futures import Future
from threading import Lock

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.weight_update.rollout_cell_updater import _RolloutCellUpdater

logger = logging.getLogger(__name__)


# This class, like the rest of the p2p weight-update code, is kept deliberately naive until yueming's refactor part 2 reshapes it.
class _P2PRolloutCellUpdater(_RolloutCellUpdater):
    def __init__(self, args: Namespace, cell_id: str, api_client: SGLangApiClient) -> None:
        super().__init__(args=args, cell_id=cell_id, api_client=api_client)
        self._pending_writes: set[Future[None]] = set()
        self._pending_writes_lock = Lock()

    def run_write(self, write: Callable[[], None]) -> None:
        if self.is_errored:
            logger.warning(f"[P2P-Shared] skipping a queued write to rollout cell {self.cell_id}")
            return
        try:
            write()
        except Exception as error:
            self.mark_errored(error)
            raise

    def track_write(self, future: Future[None]) -> None:
        with self._pending_writes_lock:
            self._pending_writes.add(future)
        future.add_done_callback(self._forget_write)
        if self.is_errored:
            future.cancel()

    def mark_errored(self, error: BaseException) -> None:
        super().mark_errored(error)
        with self._pending_writes_lock:
            pending = tuple(self._pending_writes)
        for future in pending:
            future.cancel()

    def _forget_write(self, future: Future[None]) -> None:
        with self._pending_writes_lock:
            self._pending_writes.discard(future)
