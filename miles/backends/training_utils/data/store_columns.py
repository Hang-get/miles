"""Trainer side of the output store path: execute this shard's merge plans into training columns.

Runs once per train attempt on this rank's DP shard, after the shard is selected
(``process_rollout_data``): pops the shard's ``store_plan`` / ``store_refs`` columns, reads each
(object, stored field) a segment uses once through the package's ``read_many``, executes each
field's plans with ``miles.utils.output_store.materialize`` and writes the columns the inline path's
``convert_samples_to_train_data`` writes, so nothing after this changes. The trainer never deletes
an object: the shard's other ranks and FT retries read the same objects, and the rollout executor
deletes them once their rollout is complete.
"""

from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from miles.rollout.generate_utils.rollout_topk_logprobs import validate_rollout_topk_logprobs_sample
from miles.utils.output_store.config import get_output_store_reader, output_store_read_buffer_bytes
from miles.utils.output_store.materialize import (
    check_csr_objects,
    check_nonzero_objects,
    check_support,
    compact_candidates,
    expand_rows,
    gather_csr_rows,
    gather_dense_rows,
    redirect_forced_rows,
    stack_objects,
    support_topk,
)
from miles.utils.output_store.plan import FIELDS, INDEXER_TOPK, MASK_LOGPROBS, FieldDef, StorePlan, ref_from_builtins
from miles.utils.output_store.rows import RespRows, SeqRows
from miles.utils.sampling_mask import RolloutSamplingMask
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

STORE_PLAN_COLUMN = "store_plan"
STORE_REFS_COLUMN = "store_refs"
_DENSE_COLUMNS = {"routed_experts": "rollout_routed_experts", INDEXER_TOPK: "rollout_indexer_topk"}


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
    _check_indexer_streams(args, shard)
    arrays = _read_objects(args, shard)
    for f, field in enumerate(FIELDS):
        if plans and plans[0].fields[f] is not None:
            _EXECUTORS[field.kind](args, rollout_data, field, f, shard, arrays)


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


def _check_indexer_streams(args: Namespace, shard: _ShardObjects) -> None:
    expected = getattr(args, "rollout_indexer_topk_num_streams", None)
    if expected is None:
        return
    for ref in shard.refs:
        spec = {spec.name: spec for spec in ref.fields}.get(INDEXER_TOPK)
        if spec is not None and spec.shape[1] != expected:
            raise ValueError(
                f"output store object {ref.key} has indexer_topk with {spec.shape[1]} streams but the model has "
                f"{expected} indexer layers; replaying it would map streams to the wrong layers."
            )


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


def _split(values: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    return [values[start:stop] for start, stop in zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True)]


# ============================== forward fields ==============================


def _execute_seq_field(
    args: Namespace,
    rollout_data: dict[str, Any],
    field: FieldDef,
    f: int,
    shard: _ShardObjects,
    arrays: dict[int, dict[str, np.ndarray]],
) -> None:
    (stored,) = field.stored
    objs = shard.used_objects(f)
    values, obj_row_offsets = stack_objects([arrays[obj][stored.name] for obj in objs], empty_dtype=stored.train_dtype)
    if field.check_nonzero:
        check_nonzero_objects(values, obj_row_offsets)
    index = expand_rows(*_local_segments(shard, f, objs), obj_row_offsets)
    rows = gather_dense_rows(values, index.source, out_dtype=np.dtype(stored.train_dtype))
    rollout_data[_DENSE_COLUMNS[field.name]] = _split(rows, index.sample_row_offsets)


# ============================== sampler fields ==============================


def _execute_resp_field(
    args: Namespace,
    rollout_data: dict[str, Any],
    field: FieldDef,
    f: int,
    shard: _ShardObjects,
    arrays: dict[int, dict[str, np.ndarray]],
) -> None:
    lengths_stored, ids_stored, logprobs_stored = field.stored
    objs = shard.used_objects(f)
    support_mode = args.rollout_sampling_logprobs_mode == "support"
    if support_mode:
        missing = [shard.refs[obj].key for obj in objs if MASK_LOGPROBS not in arrays[obj]]
        if missing:
            raise ValueError(
                f"support log-prob mode needs {MASK_LOGPROBS}; output store objects {missing[:4]} lack it"
            )
    value_stored = (ids_stored, logprobs_stored) if support_mode else (ids_stored,)
    lengths, obj_row_offsets = stack_objects(
        [arrays[obj][lengths_stored.name] for obj in objs], empty_dtype=lengths_stored.train_dtype
    )
    values, value_offsets = zip(
        *(stack_objects([arrays[obj][s.name] for obj in objs], empty_dtype=s.train_dtype) for s in value_stored),
        strict=True,
    )
    check_csr_objects(lengths, obj_row_offsets, value_offsets)
    index = expand_rows(*_local_segments(shard, f, objs), obj_row_offsets)
    response_tokens, sample_token_offsets = _response_tokens(rollout_data)
    if not np.array_equal(index.sample_row_offsets, sample_token_offsets):
        raise ValueError("sampling mask rows do not match the response lengths")

    # A forced row (a token Miles inserted) was not sampled: its support is the token itself, with
    # log-prob 0, which no column reads (top-k and compaction skip forced rows).
    forced = index.source < 0
    fill_ids = response_tokens[forced]
    fill = (fill_ids, np.zeros(len(fill_ids)))[: len(values)]
    lengths = np.concatenate((lengths, np.ones(len(fill_ids), dtype=lengths.dtype)))
    values = [np.concatenate((array, part.astype(array.dtype))) for array, part in zip(values, fill, strict=True)]
    row_offsets, (out_ids, *rest) = gather_csr_rows(
        lengths,
        values,
        redirect_forced_rows(index.source, fill_start=int(obj_row_offsets[-1])),
        out_dtypes=[np.dtype(s.train_dtype) for s in value_stored],
    )
    out_logprobs = rest[0] if support_mode else None
    sample_row_offsets = index.sample_row_offsets
    check_support(out_ids, row_offsets, response_tokens, exactly_once=support_mode)

    rollout_data["rollout_sampling_mask_ids"] = [
        torch.from_numpy(out_ids[row_offsets[a] : row_offsets[b]]) for a, b in _pairs(sample_row_offsets)
    ]
    rollout_data["rollout_sampling_mask_offsets"] = [
        torch.from_numpy(row_offsets[a : b + 1] - row_offsets[a]) for a, b in _pairs(sample_row_offsets)
    ]
    if (k := args.rollout_top_logprobs_num) and support_mode:
        topk_ids, topk_logprobs = support_topk(out_ids, row_offsets, out_logprobs, forced, k)
        rollout_data["rollout_topk_token_ids"] = _split(topk_ids, sample_row_offsets)
        rollout_data["rollout_topk_log_probs"] = _split(topk_logprobs, sample_row_offsets)
    if args.ci_test and k:
        _validate_topk_candidates(rollout_data, k)
    if k and args.loss_type == "score_centering" and support_mode:
        lengths_out = compact_candidates(row_offsets, forced)
        rollout_data["rollout_topk_lengths"] = [torch.from_numpy(x) for x in _split(lengths_out, sample_row_offsets)]
        del rollout_data["rollout_topk_token_ids"]


def _response_tokens(rollout_data: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lengths = [int(n) for n in rollout_data["response_lengths"]]
    parts = [
        np.asarray(tokens[len(tokens) - n :], dtype=np.int64)
        for tokens, n in zip(rollout_data["tokens"], lengths, strict=True)
    ]
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return (np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)), offsets


def _pairs(offsets: np.ndarray) -> list[tuple[int, int]]:
    return list(zip(offsets[:-1].tolist(), offsets[1:].tolist(), strict=True))


def _validate_topk_candidates(rollout_data: dict[str, Any], k: int) -> None:
    """--ci-test: the inline path's per-sample candidate check, now that the sampling mask is assembled."""
    for i, tokens in enumerate(rollout_data["tokens"]):
        sample = Sample(
            tokens=list(tokens),
            response_length=int(rollout_data["response_lengths"][i]),
            loss_mask=list(rollout_data["loss_masks"][i]),
            rollout_log_probs=list(rollout_data["rollout_log_probs"][i]),
            rollout_topk_token_ids=rollout_data["rollout_topk_token_ids"][i],
            rollout_topk_log_probs=rollout_data["rollout_topk_log_probs"][i],
            rollout_sampling_mask=RolloutSamplingMask(
                ids=rollout_data["rollout_sampling_mask_ids"][i],
                offsets=rollout_data["rollout_sampling_mask_offsets"][i],
            ),
        )
        validate_rollout_topk_logprobs_sample(sample, k)


_EXECUTORS = {SeqRows: _execute_seq_field, RespRows: _execute_resp_field}
