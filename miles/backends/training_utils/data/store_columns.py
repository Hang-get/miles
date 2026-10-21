"""Trainer side of the output store path: execute this shard's merge plans into training columns.

Runs once per train attempt on this rank's DP shard, after the shard is selected
(``process_rollout_data``): pops the shard's ``store_plan`` / ``store_refs`` columns, reads each
(object, stored field) a segment uses once through the package's ``read_many``, has each field
materialize its columns (``miles.utils.output_store.fields``) and writes the columns the inline
path's ``convert_samples_to_train_data`` writes, so nothing after this changes. The trainer never
deletes an object: the shard's other ranks and FT retries read the same objects, and the rollout
executor deletes them once their rollout is complete.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import numpy as np

from miles.utils.output_store.config import get_output_store_reader, output_store_read_buffer_bytes
from miles.utils.output_store.plan import FIELDS, StorePlan, ref_from_builtins

logger = logging.getLogger(__name__)

STORE_PLAN_COLUMN = "store_plan"
STORE_REFS_COLUMN = "store_refs"


def materialize_store_columns(args: Namespace, rollout_data: dict[str, Any]) -> None:
    """Replace this shard's store columns with the training columns; a no-op without them."""
    if STORE_PLAN_COLUMN not in rollout_data:
        return
    tables = rollout_data.pop(STORE_PLAN_COLUMN)
    refs = rollout_data.pop(STORE_REFS_COLUMN)
    plans = [
        StorePlan.from_table(np.asarray(table), [ref_from_builtins(ref) for ref in owned])
        for table, owned in zip(tables, refs, strict=True)
    ]
    shard = _ShardObjects(plans)
    arrays = _read_objects(args, shard)
    for f, field in enumerate(FIELDS):
        if plans and plans[0].fields[f] is not None:
            objs = shard.used_objects(f)
            objects = {shard.refs[obj].key: arrays[obj] for obj in objs}
            rollout_data.update(field.materialize(args, rollout_data, objects, _local_segments(shard, f, objs)))


class _ShardObjects:
    """The shard's objects, each once, and every field's segments as flat arrays."""

    def __init__(self, plans: Sequence[StorePlan]) -> None:
        self.plans = plans
        self.refs = list({ref.key: ref for plan in plans for ref in plan.owned_refs}.values())
        self.index = {ref.key: i for i, ref in enumerate(self.refs)}

    def segments(self, f: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(seg_obj, seg_start, seg_stop, sample_seg_offsets)`` of field ``f``; ``seg_obj`` is a shard index."""
        segs = [seg for plan in self.plans for seg in plan.fields[f].segs]
        counts = [len(plan.fields[f].segs) for plan in self.plans]
        seg_obj = np.asarray([-1 if seg.ref is None else self.index[seg.ref.key] for seg in segs], dtype=np.int64)
        seg_start = np.asarray([seg.start for seg in segs], dtype=np.int64)
        seg_stop = np.asarray([seg.stop for seg in segs], dtype=np.int64)
        return seg_obj, seg_start, seg_stop, np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))

    def used_objects(self, f: int) -> list[int]:
        return sorted({self.index[seg.ref.key] for plan in self.plans for seg in plan.fields[f].segs if seg.ref})


def _read_objects(args: Namespace, shard: _ShardObjects) -> dict[int, dict[str, np.ndarray]]:
    """Each used object's stored arrays this shard needs, read once and owned by this process."""
    wanted: dict[int, list[str]] = {}
    for f, field in enumerate(FIELDS):
        if not shard.plans or shard.plans[0].fields[f] is None:
            continue
        for obj in shard.used_objects(f):
            present = {spec.name for spec in shard.refs[obj].fields}
            wanted.setdefault(obj, []).extend(stored.name for stored in field.stored if stored.name in present)
    if not wanted:
        return {}

    from radix_transfer_plugins.stashes.token_output import TokenOutputStash

    objs = list(wanted)
    stashes = get_output_store_reader().read_many(
        [(shard.refs[obj], wanted[obj]) for obj in objs],
        stash_type=TokenOutputStash,
        buffer_bytes=output_store_read_buffer_bytes(args),
    )
    logger.info(
        f"Read {sum(len(names) for names in wanted.values())} output store fields of {len(objs)} objects, "
        f"{sum(spec.nbytes for obj in objs for spec in shard.refs[obj].fields if spec.name in wanted[obj])} bytes"
    )
    return {
        obj: {name: getattr(stash, name) for name in wanted[obj]} for obj, stash in zip(objs, stashes, strict=True)
    }


def _local_segments(shard: _ShardObjects, f: int, objs: list[int]) -> tuple[np.ndarray, ...]:
    """Field ``f``'s segments with ``seg_obj`` indexing ``objs``; forced segments keep -1."""
    seg_obj, seg_start, seg_stop, sample_seg_offsets = shard.segments(f)
    # The extra last slot is -1, where forced segments' -1 lands.
    local = np.full(len(shard.refs) + 1, -1, dtype=np.int64)
    local[objs] = np.arange(len(objs))
    return local[seg_obj], seg_start, seg_stop, sample_seg_offsets
