"""Materialize functions of the output store path, against per-row Python references."""

import numpy as np
import pytest

from miles.utils.output_store.materialize import (
    StoreExecError,
    check_nonzero_objects,
    check_support,
    compact_candidates,
    gather_csr_rows,
    gather_dense_rows,
    support_topk,
)


def _offsets(counts) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))


def _segments(per_sample: list[list[tuple[int, int, int]]]) -> tuple[np.ndarray, ...]:
    flat = [seg for segs in per_sample for seg in segs]
    obj, start, stop = (np.asarray([seg[i] for seg in flat], dtype=np.int64).reshape(-1) for i in range(3))
    return obj, start, stop, _offsets([len(segs) for segs in per_sample])


class TestGatherDenseRows:
    def test_concatenates_each_samples_segments_and_converts_the_dtype(self):
        objs = [np.arange(4 * 2 * 3, dtype=np.int16).reshape(4, 2, 3), 100 + np.arange(3 * 2 * 3).reshape(3, 2, 3)]
        values = np.concatenate([o.astype(np.int16) for o in objs])
        plan = [[(0, 0, 2), (1, 1, 3)], [], [(1, 0, 3)]]

        rows, offsets = gather_dense_rows(values, _offsets([4, 3]), *_segments(plan), out_dtype=np.int32)

        expected = [np.concatenate([objs[0][0:2], objs[1][1:3]]), np.empty((0, 2, 3)), objs[1][0:3]]
        assert rows.dtype == np.int32 and offsets.tolist() == [0, 4, 4, 7]
        for i, want in enumerate(expected):
            np.testing.assert_array_equal(rows[offsets[i] : offsets[i + 1]], want)

    def test_rejects_forced_rows_unsafe_dtypes_and_out_of_range_segments(self):
        values = np.ones((3, 1, 1), dtype=np.int32)
        with pytest.raises(StoreExecError, match="cannot be forced.*sample=1"):
            gather_dense_rows(values, _offsets([3]), *_segments([[(0, 0, 1)], [(-1, 0, 2)]]), out_dtype=np.int32)
        with pytest.raises(StoreExecError, match="out of range.*sample=0"):
            gather_dense_rows(values, _offsets([3]), *_segments([[(0, 1, 4)]]), out_dtype=np.int32)
        with pytest.raises(StoreExecError, match="convert safely"):
            gather_dense_rows(values.astype(np.int64), _offsets([3]), *_segments([[(0, 0, 1)]]), out_dtype=np.int32)


def _csr(supports: list[list[int]], logprobs: list[list[float]] | None = None):
    lengths = np.asarray([len(s) for s in supports], dtype=np.int64)
    ids = np.asarray([t for s in supports for t in s], dtype=np.int32)
    lps = None if logprobs is None else np.asarray([v for r in logprobs for v in r], dtype=np.float32)
    return lengths, ids, lps


class TestGatherCsrRows:
    def test_slices_rows_and_fills_forced_rows_from_the_response(self):
        obj0 = ([[5, 6], [7], [8, 9, 10]], [[-0.1, -0.2], [-0.3], [-0.4, -0.5, -0.6]])
        obj1 = ([[11], [12, 13]], [[-0.7], [-0.8, -0.9]])
        lengths, ids, lps = (np.concatenate(parts) for parts in zip(_csr(*obj0), _csr(*obj1), strict=True))
        # Sample 0: rows 1..3 of obj0, two forced rows, row 0 of obj1. Sample 1: rows 0..2 of obj1.
        plan = [[(0, 1, 3), (-1, 0, 2), (1, 0, 1)], [(1, 0, 2)]]
        response = [[7, 8, 40, 41, 11], [11, 12]]

        out_ids, row_offsets, sample_rows, out_lps, forced = gather_csr_rows(
            lengths,
            ids,
            lps,
            _offsets([3, 2]),
            _offsets([6, 3]),
            *_segments(plan),
            np.asarray([t for r in response for t in r]),
            _offsets([len(r) for r in response]),
        )

        supports = [obj0[0][1], obj0[0][2], [40], [41], obj1[0][0], obj1[0][0], obj1[0][1]]
        logprobs = [obj0[1][1], obj0[1][2], [0.0], [0.0], obj1[1][0], obj1[1][0], obj1[1][1]]
        assert sample_rows.tolist() == [0, 5, 7]
        assert forced.tolist() == [False, False, True, True, False, False, False]
        assert [out_ids[a:b].tolist() for a, b in zip(row_offsets[:-1], row_offsets[1:])] == supports
        np.testing.assert_array_equal(out_lps, np.asarray([v for r in logprobs for v in r], dtype=np.float32))
        assert out_ids.dtype == np.int32 and row_offsets.dtype == np.int64

    def test_only_forced_rows_need_no_object(self):
        out_ids, row_offsets, sample_rows, out_lps, forced = gather_csr_rows(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int32),
            None,
            _offsets([]),
            _offsets([]),
            *_segments([[(-1, 0, 2)], []]),
            np.asarray([3, 4]),
            _offsets([2, 0]),
        )
        assert out_ids.tolist() == [3, 4] and row_offsets.tolist() == [0, 1, 2] and out_lps is None
        assert sample_rows.tolist() == [0, 2, 2] and forced.all()

    def test_rejects_inconsistent_objects_and_forced_rows_past_the_response(self):
        lengths, ids, _ = _csr([[1, 2], [3]])
        with pytest.raises(StoreExecError, match="do not sum"):
            gather_csr_rows(
                lengths, ids, None, _offsets([1, 1]), _offsets([1, 2]), *_segments([[(0, 0, 1), (1, 0, 1)]]),
                np.asarray([1, 3]), _offsets([2]),
            )  # fmt: skip
        with pytest.raises(StoreExecError, match="past the response.*sample=0, row=2"):
            gather_csr_rows(
                lengths, ids, None, _offsets([2]), _offsets([3]), *_segments([[(0, 0, 2), (-1, 0, 1)]]),
                np.asarray([1, 3]), _offsets([2]),
            )  # fmt: skip


class TestSupportChecks:
    def test_sampled_token_must_be_in_its_support(self):
        ids = np.asarray([5, 6, 7, 7, 8], dtype=np.int32)
        offsets = np.asarray([0, 2, 3, 5])
        check_support(ids, offsets, np.asarray([6, 7, 8]), exactly_once=True)
        with pytest.raises(StoreExecError, match="absent.*row=1"):
            check_support(ids, offsets, np.asarray([6, 9, 8]), exactly_once=False)
        with pytest.raises(StoreExecError, match="exactly once.*row=2"):
            check_support(np.asarray([5, 6, 7, 8, 8]), offsets, np.asarray([6, 7, 8]), exactly_once=True)
        with pytest.raises(StoreExecError, match="at least one token.*row=1"):
            check_support(ids[:2], np.asarray([0, 2, 2]), np.asarray([5, 1]), exactly_once=False)

    def test_all_zero_objects_are_rejected(self):
        values = np.zeros((5, 1, 2), dtype=np.int32)
        values[1, 0, 1] = 3
        check_nonzero_objects(values[:2], np.asarray([0, 2, 2]))
        with pytest.raises(StoreExecError, match="all zeros.*object 2"):
            check_nonzero_objects(values, np.asarray([0, 2, 2, 5]))


class TestSupportTopk:
    def test_lays_out_each_support_with_forced_and_unused_slots_empty(self):
        ids = np.asarray([5, 6, 9, 7], dtype=np.int32)
        offsets = np.asarray([0, 2, 3, 4])
        logprobs = np.asarray([-0.1, -0.2, 0.0, -0.3], dtype=np.float32)
        forced = np.asarray([False, True, False])

        topk_ids, topk_lps = support_topk(ids, offsets, logprobs, forced, 3)

        assert topk_ids.tolist() == [[5, 6, -1], [-1, -1, -1], [7, -1, -1]]
        np.testing.assert_array_equal(topk_lps[0], np.asarray([-0.1, -0.2, -np.inf], dtype=np.float32))
        assert np.isneginf(topk_lps[1]).all()
        assert compact_candidates(offsets, forced).tolist() == [2, 0, 1]

    def test_support_wider_than_k_is_rejected(self):
        with pytest.raises(StoreExecError, match="exceeds --rollout-top-logprobs-num.*row=0"):
            support_topk(
                np.arange(3, dtype=np.int32), np.asarray([0, 3]), np.zeros(3, np.float32), np.asarray([False]), 2
            )
