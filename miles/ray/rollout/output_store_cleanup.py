"""The rollout executor's deletion of output store objects, in batches per completed rollout.

Objects are never deleted online or by the trainer: every rank of a shard reads the same objects,
FT retries read them again, and the output snapshot may replay a batch until its rollout is
completed. The executor files each converted batch's owned refs, plus the refs of samples dropped
before conversion, under the rollout id, and deletes them where the snapshot discards completed
rollouts.
"""

import logging
from argparse import Namespace
from typing import Any

from miles.utils.output_store_config import connect_output_store
from miles.utils.store_plan import owned_refs_of_samples, take_dropped_refs

logger = logging.getLogger(__name__)


class OutputStoreCleanup:
    def __init__(self, *, args: Namespace) -> None:
        self._args = args
        self._store = None
        # (trainer model id, rollout id) -> store key -> ref
        self._pending: dict[tuple[str | None, int], dict[str, Any]] = {}

    def connect(self) -> None:
        """Connect this process's client; a no-op without --use-output-store."""
        if self._args.use_output_store:
            self._store = connect_output_store(self._args)

    def record(self, *, trainer_model_id: str | None, rollout_id: int, samples: list[Any]) -> None:
        """File a converted batch's refs, and those dropped since the last batch, under its rollout."""
        if self._store is None:
            return
        refs = owned_refs_of_samples(samples) + take_dropped_refs()
        if refs:
            self._pending.setdefault((trainer_model_id, rollout_id), {}).update((ref.key, ref) for ref in refs)

    def release_completed(self, *, completed_rollout_ids: dict[str | None, int]) -> None:
        """Delete the objects of every rollout up to each model's completed one."""
        if self._store is None:
            return
        done = [
            key for key in self._pending if key[0] in completed_rollout_ids and key[1] <= completed_rollout_ids[key[0]]
        ]
        refs = {store_key: ref for key in done for store_key, ref in self._pending.pop(key).items()}
        if not refs:
            return
        try:
            self._store.delete_many(list(refs.values()))
        except Exception:
            # Not retried: a missed object only stays until the storage plane is torn down.
            logger.exception(f"Failed to delete {len(refs)} output store objects (first keys {list(refs)[:4]})")
            return
        logger.info(f"Deleted {len(refs)} output store objects of completed rollouts {sorted(key[1] for key in done)}")
