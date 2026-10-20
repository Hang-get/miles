"""Materialize step of the output store path: pure functions over flat numpy arrays.

The trainer executes a shard's merge plans (``miles.utils.output_store.plan``) with these. Conventions,
so each function can be rewritten in another language:

- inputs are contiguous numpy arrays, of a dtype from a small fixed set, plus ints;
- batch-first: one call handles every sample of a shard, with no per-element Python loop;
- outputs are newly allocated arrays of the training columns' dtypes;
- the only error is ``StoreExecError``, which names the sample and row it is about.

Layout of one shard (``obj`` = a stored object, ``seg`` = a segment of a plan):

- object level, per stored field: the arrays of every object concatenated, with
  ``obj_row_offsets`` ``[num_objs + 1]``; CSR fields add ``obj_entry_offsets``;
- plan level: ``seg_obj``, ``seg_start``, ``seg_stop`` ``[num_segs]`` int32 (``seg_obj = -1``
  for forced rows) with ``sample_seg_offsets`` ``[num_samples + 1]``;
- forced rows' tokens: ``response_tokens`` (flat) with ``sample_token_offsets``.
"""

from __future__ import annotations

import numpy as np


class StoreExecError(ValueError):
    """Replay data from the output store is inconsistent with its plan."""

    def __init__(self, message: str, *, sample: int | None = None, row: int | None = None) -> None:
        where = ", ".join(f"{name}={value}" for name, value in (("sample", sample), ("row", row)) if value is not None)
        super().__init__(f"{message} ({where})" if where else message)
        self.sample = sample
        self.row = row


def gather_dense_rows(
    values: np.ndarray,
    obj_row_offsets: np.ndarray,
    seg_obj: np.ndarray,
    seg_start: np.ndarray,
    seg_stop: np.ndarray,
    sample_seg_offsets: np.ndarray,
    out_dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray]:
    """Each sample's rows of a forward field, concatenated in plan order.

    ``values`` is ``[num_obj_rows, ...]`` of any stored dtype that converts safely to
    ``out_dtype``; the conversion happens here only. Returns ``(rows, sample_row_offsets)``.
    """
    if not np.can_cast(values.dtype, out_dtype, casting="safe"):
        raise StoreExecError(f"stored dtype {values.dtype} does not convert safely to {np.dtype(out_dtype)}")
    seg_sample = _seg_samples(sample_seg_offsets)
    forced = seg_obj < 0
    if forced.any():
        raise StoreExecError("forward rows cannot be forced", sample=int(seg_sample[np.argmax(forced)]))
    _check_segments(seg_obj, seg_start, seg_stop, obj_row_offsets, seg_sample)
    lengths = (seg_stop - seg_start).astype(np.int64)
    within = _ragged_arange(lengths)
    source = np.repeat(obj_row_offsets[seg_obj] + seg_start, lengths) + within
    rows = np.take(values, source, axis=0).astype(out_dtype, copy=False)
    return rows, _sample_offsets(lengths, sample_seg_offsets)


def gather_csr_rows(
    lengths: np.ndarray,
    ids: np.ndarray,
    logprobs: np.ndarray | None,
    obj_row_offsets: np.ndarray,
    obj_entry_offsets: np.ndarray,
    seg_obj: np.ndarray,
    seg_start: np.ndarray,
    seg_stop: np.ndarray,
    sample_seg_offsets: np.ndarray,
    response_tokens: np.ndarray,
    sample_token_offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Each sample's rows of a sampler field (a CSR support per row), in plan order.

    A forced row's support is the token at its response position alone, with log-prob 0;
    forced rows are filled here only. Returns ``(ids, row_offsets, sample_row_offsets,
    logprobs, forced)``: ``row_offsets`` ``[num_rows + 1]`` int64 index ``ids`` and
    ``logprobs`` (float32, None when ``logprobs`` is None); ``forced`` is ``[num_rows]`` bool.
    """
    seg_sample = _seg_samples(sample_seg_offsets)
    entry_starts = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=entry_starts[1:])
    _check_csr_objects(lengths, ids, logprobs, obj_row_offsets, obj_entry_offsets, entry_starts)
    real = seg_obj >= 0
    _check_segments(seg_obj[real], seg_start[real], seg_stop[real], obj_row_offsets, seg_sample[real])

    seg_rows = (seg_stop - seg_start).astype(np.int64)
    sample_row_offsets = _sample_offsets(seg_rows, sample_seg_offsets)
    row_seg = np.repeat(np.arange(len(seg_rows)), seg_rows)
    forced = ~real[row_seg]
    row_sample = seg_sample[row_seg]
    position = np.arange(len(row_seg)) - sample_row_offsets[row_sample]
    forced_token = _forced_tokens(forced, row_sample, position, response_tokens, sample_token_offsets)

    source_row = np.where(forced, 0, np.repeat(obj_row_offsets[np.maximum(seg_obj, 0)] + seg_start, seg_rows))
    source_row = source_row + _ragged_arange(seg_rows) * ~forced
    row_lengths = np.where(forced, 1, lengths[source_row] if len(lengths) else 0).astype(np.int64)
    row_offsets = np.zeros(len(row_lengths) + 1, dtype=np.int64)
    np.cumsum(row_lengths, out=row_offsets[1:])

    entry_row = np.repeat(np.arange(len(row_lengths)), row_lengths)
    entry_forced = forced[entry_row]
    source_entry = entry_starts[source_row][entry_row] + (np.arange(len(entry_row)) - row_offsets[entry_row])
    out_ids = np.empty(len(entry_row), dtype=np.int32)
    out_ids[entry_forced] = forced_token[entry_row[entry_forced]]
    out_ids[~entry_forced] = ids[source_entry[~entry_forced]]
    out_logprobs = None
    if logprobs is not None:
        out_logprobs = np.zeros(len(entry_row), dtype=np.float32)
        out_logprobs[~entry_forced] = logprobs[source_entry[~entry_forced]]
    return out_ids, row_offsets, sample_row_offsets, out_logprobs, forced


def check_nonzero_objects(values: np.ndarray, obj_row_offsets: np.ndarray) -> None:
    """Every object with rows has a nonzero value; an all-zero one was never captured by the engine."""
    nonzero_rows = (values.reshape(len(values), -1) != 0).any(axis=1)
    counts = np.zeros(len(values) + 1, dtype=np.int64)
    np.cumsum(nonzero_rows, out=counts[1:])
    empty = (np.diff(obj_row_offsets) > 0) & (counts[obj_row_offsets[1:]] == counts[obj_row_offsets[:-1]])
    if empty.any():
        raise StoreExecError(
            "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
            f"(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?); object {int(np.argmax(empty))}"
        )


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


def _seg_samples(sample_seg_offsets: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(sample_seg_offsets) - 1), np.diff(sample_seg_offsets))


def _ragged_arange(lengths: np.ndarray) -> np.ndarray:
    """``concatenate([arange(n) for n in lengths])`` without a loop."""
    starts = np.zeros(len(lengths), dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    return np.arange(int(lengths.sum()), dtype=np.int64) - np.repeat(starts, lengths)


def _sample_offsets(seg_rows: np.ndarray, sample_seg_offsets: np.ndarray) -> np.ndarray:
    seg_row_offsets = np.zeros(len(seg_rows) + 1, dtype=np.int64)
    np.cumsum(seg_rows, out=seg_row_offsets[1:])
    return seg_row_offsets[sample_seg_offsets]


def _check_segments(
    seg_obj: np.ndarray,
    seg_start: np.ndarray,
    seg_stop: np.ndarray,
    obj_row_offsets: np.ndarray,
    seg_sample: np.ndarray,
) -> None:
    num_objs = len(obj_row_offsets) - 1
    bad = (seg_obj >= num_objs) | (seg_start < 0) | (seg_start > seg_stop)
    if not bad.any():
        obj_rows = np.diff(obj_row_offsets)[seg_obj]
        bad = seg_stop > obj_rows
    if bad.any():
        i = int(np.argmax(bad))
        raise StoreExecError(
            f"segment rows [{int(seg_start[i])}, {int(seg_stop[i])}) of object {int(seg_obj[i])} are out of range",
            sample=int(seg_sample[i]),
        )


def _check_csr_objects(
    lengths: np.ndarray,
    ids: np.ndarray,
    logprobs: np.ndarray | None,
    obj_row_offsets: np.ndarray,
    obj_entry_offsets: np.ndarray,
    entry_starts: np.ndarray,
) -> None:
    if obj_row_offsets[-1] != len(lengths) or obj_entry_offsets[-1] != len(ids):
        raise StoreExecError("CSR object offsets do not match the concatenated arrays")
    if (lengths < 0).any():
        raise StoreExecError("negative sampling support length", row=int(np.argmax(lengths < 0)))
    if not np.array_equal(entry_starts[obj_row_offsets], obj_entry_offsets):
        raise StoreExecError("an object's sampling support lengths do not sum to its token id count")
    if logprobs is not None and len(logprobs) != len(ids):
        raise StoreExecError(f"{len(logprobs)} support log-probs for {len(ids)} support token ids")


def _forced_tokens(
    forced: np.ndarray,
    row_sample: np.ndarray,
    position: np.ndarray,
    response_tokens: np.ndarray,
    sample_token_offsets: np.ndarray,
) -> np.ndarray:
    response_lengths = np.diff(sample_token_offsets)
    outside = forced & (position >= response_lengths[row_sample])
    if outside.any():
        i = int(np.argmax(outside))
        raise StoreExecError("forced row past the response", sample=int(row_sample[i]), row=int(position[i]))
    index = np.where(forced, sample_token_offsets[row_sample] + position, 0)
    if not len(response_tokens):
        return np.zeros(len(forced), dtype=np.int32)
    return response_tokens[index].astype(np.int32, copy=False)
