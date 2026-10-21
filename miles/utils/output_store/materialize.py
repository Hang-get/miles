"""Materialize step of the output store path: pure functions over flat numpy arrays.

The trainer executes a shard's merge plans (``miles.utils.output_store.plan``) with these. Conventions,
so each function can be rewritten in another language:

- inputs are contiguous numpy arrays, of a dtype from a small fixed set, plus ints;
- batch-first: one call handles every sample of a shard, with no per-element Python loop;
- outputs are newly allocated arrays of the training columns' dtypes;
- the only error is ``StoreExecError``, which names the sample and row it is about.

Layout of one shard (``obj`` = a stored object, ``seg`` = a segment of a plan):

- object level, per stored field: the arrays of every object concatenated (``stack_objects``),
  with ``obj_row_offsets`` ``[num_objs + 1]``; a CSR field's values arrays have their own offsets;
- plan level: ``seg_obj``, ``seg_start``, ``seg_stop`` ``[num_segs]`` (``seg_obj = -1`` for forced
  rows) with ``sample_seg_offsets`` ``[num_samples + 1]``;
- row level: ``RowIndex``, each row's source row in the concatenated objects.

Three steps turn them into a field's rows: ``expand_rows`` lists where each row comes from; the
caller decides what a forced row holds, appends it to the objects as fill rows and points the
forced rows at them (``redirect_forced_rows``); ``gather_dense_rows`` / ``gather_csr_rows`` copy
the listed rows, knowing nothing of forced rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np


class StoreExecError(ValueError):
    """Replay data from the output store is inconsistent with its plan."""

    def __init__(self, message: str, *, sample: int | None = None, row: int | None = None) -> None:
        where = ", ".join(f"{name}={value}" for name, value in (("sample", sample), ("row", row)) if value is not None)
        super().__init__(f"{message} ({where})" if where else message)
        self.sample = sample
        self.row = row


class RowIndex(NamedTuple):
    """Each row of a shard's plans, in plan order."""

    # [num_rows] int64: the row of the concatenated objects it copies; -1 for a forced row.
    source: np.ndarray
    # [num_samples + 1] int64: sample i's rows are source[sample_row_offsets[i]:sample_row_offsets[i + 1]].
    sample_row_offsets: np.ndarray


def stack_objects(parts: Sequence[np.ndarray], empty_dtype: np.dtype) -> tuple[np.ndarray, np.ndarray]:
    """One stored array of every object concatenated, with its ``[num_objs + 1]`` offsets."""
    offsets = np.zeros(len(parts) + 1, dtype=np.int64)
    np.cumsum([len(part) for part in parts], out=offsets[1:])
    return (np.concatenate(parts) if parts else np.empty(0, dtype=empty_dtype)), offsets


def expand_rows(
    seg_obj: np.ndarray,
    seg_start: np.ndarray,
    seg_stop: np.ndarray,
    sample_seg_offsets: np.ndarray,
    obj_row_offsets: np.ndarray,
) -> RowIndex:
    """Where each row of the shard's plans comes from; a forced segment's rows get source -1."""
    seg_sample = np.repeat(np.arange(len(sample_seg_offsets) - 1), np.diff(sample_seg_offsets))
    _check_segments(seg_obj, seg_start, seg_stop, obj_row_offsets, seg_sample)
    seg_rows = (seg_stop - seg_start).astype(np.int64)
    real = seg_obj >= 0
    # The extra last slot is where a forced segment's -1 lands.
    seg_first = np.where(real, np.append(obj_row_offsets[:-1], 0)[seg_obj] + seg_start, -1)
    source = np.repeat(seg_first, seg_rows) + _ragged_arange(seg_rows) * np.repeat(real, seg_rows)
    seg_row_offsets = np.zeros(len(seg_rows) + 1, dtype=np.int64)
    np.cumsum(seg_rows, out=seg_row_offsets[1:])
    return RowIndex(source=source, sample_row_offsets=seg_row_offsets[sample_seg_offsets])


def redirect_forced_rows(source: np.ndarray, fill_start: int) -> np.ndarray:
    """Point the forced rows, in order, at fill rows appended to the objects from ``fill_start`` on."""
    forced = source < 0
    out = source.copy()
    out[forced] = fill_start + np.arange(int(forced.sum()))
    return out


def gather_dense_rows(values: np.ndarray, source: np.ndarray, out_dtype: np.dtype) -> np.ndarray:
    """``values[source]`` ``[num_rows, ...]``, converted to ``out_dtype``.

    ``values`` may be of any stored dtype that converts safely to ``out_dtype``; the conversion
    happens here only.
    """
    _check_cast(values.dtype, out_dtype)
    _check_sources(source, len(values))
    return np.take(values, source, axis=0).astype(out_dtype, copy=False)


def gather_csr_rows(
    lengths: np.ndarray, values: Sequence[np.ndarray], source: np.ndarray, out_dtypes: Sequence[np.dtype]
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """The CSR rows ``source`` lists: each values array's entries of those rows, in order.

    ``lengths`` ``[num_obj_rows]`` counts each object row's entries; every values array holds
    ``lengths.sum()`` entries. Returns ``(row_offsets, values)``: ``row_offsets`` ``[num_rows + 1]``
    int64 indexes each output values array, converted to its ``out_dtypes``.
    """
    for array, dtype in zip(values, out_dtypes, strict=True):
        _check_cast(array.dtype, dtype)
    _check_sources(source, len(lengths))
    entry_starts = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=entry_starts[1:])
    row_lengths = lengths[source].astype(np.int64)
    row_offsets = np.zeros(len(source) + 1, dtype=np.int64)
    np.cumsum(row_lengths, out=row_offsets[1:])
    entry = np.repeat(entry_starts[source], row_lengths) + _ragged_arange(row_lengths)
    return row_offsets, tuple(
        array[entry].astype(dtype, copy=False) for array, dtype in zip(values, out_dtypes, strict=True)
    )


def check_csr_objects(lengths: np.ndarray, obj_row_offsets: np.ndarray, value_offsets: Sequence[np.ndarray]) -> None:
    """Each object's support lengths sum to the entry count of each of its values arrays."""
    if (lengths < 0).any():
        raise StoreExecError("negative sampling support length", row=int(np.argmax(lengths < 0)))
    entry_starts = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=entry_starts[1:])
    if any(not np.array_equal(entry_starts[obj_row_offsets], offsets) for offsets in value_offsets):
        raise StoreExecError("an object's sampling support lengths do not sum to its entry count")


def check_support(ids: np.ndarray, row_offsets: np.ndarray, sampled_tokens: np.ndarray, *, exactly_once: bool) -> None:
    """Every support is non-empty and holds its sampled token (exactly once in support mode)."""
    row_lengths = np.diff(row_offsets)
    if (empty := row_lengths == 0).any():
        raise StoreExecError("sampling support must contain at least one token", row=int(np.argmax(empty)))
    entry_row = np.repeat(np.arange(len(row_lengths)), row_lengths)
    hits = np.bincount(entry_row[ids == sampled_tokens[entry_row]], minlength=len(row_lengths))
    bad = hits != 1 if exactly_once else hits == 0
    if bad.any():
        row = int(np.argmax(bad))
        what = "must occur exactly once in" if exactly_once else "is absent from"
        raise StoreExecError(f"sampled token {int(sampled_tokens[row])} {what} its sampling support", row=row)


def support_topk(
    ids: np.ndarray, row_offsets: np.ndarray, logprobs: np.ndarray, forced: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """Support-mode top-k candidates ``[num_rows, k]``: each row's support and its log-probs.

    Unused slots and forced rows hold -1 / -inf.
    """
    row_lengths = np.diff(row_offsets)
    if (too_long := (row_lengths > k) & ~forced).any():
        raise StoreExecError(
            "Rollout top-k logprobs sampling support exceeds --rollout-top-logprobs-num", row=int(np.argmax(too_long))
        )
    out_ids = np.full((len(row_lengths), k), -1, dtype=np.int32)
    out_logprobs = np.full((len(row_lengths), k), -np.inf, dtype=np.float32)
    entry_row = np.repeat(np.arange(len(row_lengths)), row_lengths)
    entry_col = np.arange(len(entry_row)) - row_offsets[entry_row]
    keep = ~forced[entry_row]
    out_ids[entry_row[keep], entry_col[keep]] = ids[keep]
    out_logprobs[entry_row[keep], entry_col[keep]] = logprobs[keep]
    return out_ids, out_logprobs


def compact_candidates(row_offsets: np.ndarray, forced: np.ndarray) -> np.ndarray:
    """Score-centering ``rollout_topk_lengths`` of support-mode candidates: a row's support size,
    0 for forced rows. Candidates built by ``support_topk`` are each row's support in order,
    so the support ids stand in for them."""
    return np.where(forced, 0, np.diff(row_offsets)).astype(np.int32)


# ============================== helpers ==============================


def _ragged_arange(lengths: np.ndarray) -> np.ndarray:
    """``concatenate([arange(n) for n in lengths])`` without a loop."""
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    return np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(starts, lengths)


def _check_segments(
    seg_obj: np.ndarray,
    seg_start: np.ndarray,
    seg_stop: np.ndarray,
    obj_row_offsets: np.ndarray,
    seg_sample: np.ndarray,
) -> None:
    num_objs = len(obj_row_offsets) - 1
    bad = (seg_obj < -1) | (seg_obj >= num_objs) | (seg_start < 0) | (seg_start > seg_stop)
    if not bad.any():
        # A forced segment (-1) lands on the extra last slot, which bounds nothing.
        obj_rows = np.append(np.diff(obj_row_offsets), np.iinfo(np.int64).max)[seg_obj]
        bad = seg_stop > obj_rows
    if bad.any():
        i = int(np.argmax(bad))
        raise StoreExecError(
            f"segment rows [{int(seg_start[i])}, {int(seg_stop[i])}) of object {int(seg_obj[i])} are out of range",
            sample=int(seg_sample[i]),
        )


def _check_sources(source: np.ndarray, num_rows: int) -> None:
    # np.take would wrap a negative index around to the last row.
    if (bad := (source < 0) | (source >= num_rows)).any():
        raise StoreExecError("row has no source: a forced row the field did not fill", row=int(np.argmax(bad)))


def _check_cast(dtype: np.dtype, out_dtype: np.dtype) -> None:
    if not np.can_cast(dtype, out_dtype, casting="safe"):
        raise StoreExecError(f"stored dtype {dtype} does not convert safely to {np.dtype(out_dtype)}")
