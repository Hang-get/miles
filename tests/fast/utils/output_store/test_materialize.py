"""Materialize functions of the output store path, against per-row Python references."""

import numpy as np
import pytest

from miles.utils.output_store.materialize import (
    StoreExecError,
    check_csr_objects,
    check_support,
    compact_candidates,
    expand_rows,
    gather_csr_rows,
    gather_dense_rows,
    redirect_forced_rows,
    stack_objects,
    support_topk,
)


def _offsets(counts) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))


def _segments(per_sample: list[list[tuple[int, int, int]]]) -> tuple[np.ndarray, ...]:
    flat = [seg for segs in per_sample for seg in segs]
    obj, start, stop = (np.asarray([seg[i] for seg in flat], dtype=np.int64).reshape(-1) for i in range(3))
    return obj, start, stop, _offsets([len(segs) for segs in per_sample])


def _expected_source(per_sample, obj_rows: list[int]) -> list[list[int]]:
    """Per-row reference of expand_rows."""
    first = _offsets(obj_rows)
    return [
        [-1 if obj < 0 else int(first[obj]) + row for obj, start, stop in segs for row in range(start, stop)]
        for segs in per_sample
    ]


class TestExpandRows:
    def test_lists_each_rows_source_in_plan_order_with_forced_rows_as_minus_one(self):
        plan = [[(0, 1, 3), (-1, 0, 2), (1, 0, 1)], [], [(1, 0, 2), (0, 0, 1)]]

        index = expand_rows(*_segments(plan), obj_row_offsets=_offsets([3, 2]))

        expected = _expected_source(plan, [3, 2])
        assert index.sample_row_offsets.tolist() == _offsets([len(rows) for rows in expected]).tolist()
        assert index.source.tolist() == [row for rows in expected for row in rows]

    def test_only_forced_rows_need_no_object(self):
        index = expand_rows(*_segments([[(-1, 0, 2)], []]), obj_row_offsets=_offsets([]))
        assert index.source.tolist() == [-1, -1] and index.sample_row_offsets.tolist() == [0, 2, 2]

    def test_rejects_segments_out_of_range(self):
        with pytest.raises(StoreExecError, match=r"rows \[1, 4\) of object 0 are out of range.*sample=1"):
            expand_rows(*_segments([[(0, 0, 1)], [(0, 1, 4)]]), obj_row_offsets=_offsets([3]))
        with pytest.raises(StoreExecError, match="of object 1 are out of range"):
            expand_rows(*_segments([[(1, 0, 1)]]), obj_row_offsets=_offsets([3]))
        with pytest.raises(StoreExecError, match=r"rows \[2, 1\)"):
            expand_rows(*_segments([[(-1, 2, 1)]]), obj_row_offsets=_offsets([3]))


class TestGatherDenseRows:
    def test_copies_the_listed_rows_and_converts_the_dtype(self):
        objs = [np.arange(4 * 2 * 3).reshape(4, 2, 3), 100 + np.arange(3 * 2 * 3).reshape(3, 2, 3)]
        values, obj_row_offsets = stack_objects([o.astype(np.int16) for o in objs], empty_dtype=np.int16)
        source = np.asarray([0, 1, 5, 6, 4], dtype=np.int64)

        rows = gather_dense_rows(values, source, out_dtype=np.int32)

        assert obj_row_offsets.tolist() == [0, 4, 7] and rows.dtype == np.int32
        np.testing.assert_array_equal(rows, np.concatenate(objs)[source])

    def test_rejects_rows_without_a_source_and_unsafe_dtypes(self):
        values = np.ones((3, 1, 1), dtype=np.int32)
        with pytest.raises(StoreExecError, match="no source.*row=1"):
            gather_dense_rows(values, np.asarray([0, -1]), out_dtype=np.int32)
        with pytest.raises(StoreExecError, match="convert safely"):
            gather_dense_rows(values.astype(np.int64), np.asarray([0]), out_dtype=np.int32)


def _csr(supports: list[list[int]], logprobs: list[list[float]]):
    lengths = np.asarray([len(s) for s in supports], dtype=np.int32)
    ids = np.asarray([t for s in supports for t in s], dtype=np.int32)
    lps = np.asarray([v for r in logprobs for v in r], dtype=np.float32)
    return lengths, ids, lps


class TestGatherCsrRows:
    def test_copies_the_listed_rows_including_fill_rows_appended_after_the_objects(self):
        obj0 = ([[5, 6], [7], [8, 9, 10]], [[-0.1, -0.2], [-0.3], [-0.4, -0.5, -0.6]])
        obj1 = ([[11], [12, 13]], [[-0.7], [-0.8, -0.9]])
        fill = ([[40], [41, 42]], [[0.0], [0.5, 0.6]])
        lengths, ids, lps = (
            np.concatenate(parts) for parts in zip(_csr(*obj0), _csr(*obj1), _csr(*fill), strict=True)
        )
        index = expand_rows(*_segments([[(0, 1, 3), (-1, 0, 2), (1, 0, 1)], [(1, 0, 2)]]), _offsets([3, 2]))
        source = redirect_forced_rows(index.source, fill_start=5)

        row_offsets, (out_ids, out_lps) = gather_csr_rows(
            lengths, (ids, lps), source, out_dtypes=(np.int32, np.float32)
        )

        rows = [obj0, obj1, fill]
        supports = [row for obj in rows for row in obj[0]]
        logprobs = [row for obj in rows for row in obj[1]]
        assert source.tolist() == [1, 2, 5, 6, 3, 3, 4]
        assert [out_ids[a:b].tolist() for a, b in zip(row_offsets[:-1], row_offsets[1:], strict=True)] == [
            supports[s] for s in source
        ]
        np.testing.assert_array_equal(out_lps, np.asarray([v for s in source for v in logprobs[s]], dtype=np.float32))
        assert out_ids.dtype == np.int32 and row_offsets.dtype == np.int64

    def test_rejects_rows_without_a_source(self):
        lengths, ids, _ = _csr([[1, 2], [3]], [[0, 0], [0]])
        with pytest.raises(StoreExecError, match="no source.*row=0"):
            gather_csr_rows(lengths, (ids,), np.asarray([-1, 0]), out_dtypes=(np.int32,))

    def test_objects_must_agree_on_their_entries(self):
        lengths, ids, _ = _csr([[1, 2], [3]], [[0, 0], [0]])
        check_csr_objects(lengths, _offsets([1, 1]), [_offsets([2, 1])])
        with pytest.raises(StoreExecError, match="do not sum"):
            check_csr_objects(lengths, _offsets([1, 1]), [_offsets([2, 1]), _offsets([1, 2])])
        with pytest.raises(StoreExecError, match="negative.*row=1"):
            check_csr_objects(np.asarray([1, -1]), _offsets([2]), [_offsets([0])])


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
