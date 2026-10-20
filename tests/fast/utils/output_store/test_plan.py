"""StorePlan edits: refs and row numbers only, never data. Fake refs stand in for the package's StoreRef."""

from typing import NamedTuple

import numpy as np
import pytest

from miles.utils.output_store.plan import (
    INDEXER_TOPK,
    MASK_LENGTHS,
    MASK_LOGPROBS,
    MASK_TOKEN_IDS,
    ROUTED_EXPERTS,
    SAMPLING_MASK,
    StorePlan,
    StorePlanError,
    combine_plans,
    register_dropped_samples,
    take_dropped_refs,
)
from miles.utils.output_store.rows import RespRows, Seg, SeqRows
from miles.utils.types import Sample


class Spec(NamedTuple):
    name: str
    dtype: str
    shape: tuple[int, ...]


class Ref(NamedTuple):
    key: str
    fields: tuple[Spec, ...]


def ref(
    key: str, *, seq_rows: int | None = None, mask_rows: int | None = None, support: bool = False, **dtypes
) -> Ref:
    specs = []
    if seq_rows is not None:
        specs.append(Spec(ROUTED_EXPERTS, dtypes.get("r3", "int32"), (seq_rows, 2, 4)))
        specs.append(Spec(INDEXER_TOPK, "int32", (seq_rows, 2, 8)))
    if mask_rows is not None:
        specs.append(Spec(MASK_LENGTHS, "int32", (mask_rows,)))
        specs.append(Spec(MASK_TOKEN_IDS, "int32", (3 * mask_rows,)))
        if support:
            specs.append(Spec(MASK_LOGPROBS, "float32", (dtypes.get("logprob_entries", 3 * mask_rows),)))
    return Ref(key, tuple(specs))


ALL = {ROUTED_EXPERTS, INDEXER_TOPK, SAMPLING_MASK}


def response(key: str, *, prompt: int, output: int, requested=ALL, **kwargs) -> StorePlan:
    return StorePlan.from_response(
        ref(key, seq_rows=prompt + output - 1, mask_rows=output),
        requested=requested,
        num_tokens=prompt + output,
        response_length=output,
        **kwargs,
    )


def segs(plan: StorePlan, name: str) -> list[tuple[str | None, int, int]]:
    return [(None if s.ref is None else s.ref.key, s.start, s.stop) for s in plan.field(name).segs]


class TestFromResponse:
    def test_requested_fields_take_every_row_of_the_object(self):
        plan = response("a", prompt=3, output=2)
        assert segs(plan, ROUTED_EXPERTS) == [("a", 0, 4)]
        assert segs(plan, SAMPLING_MASK) == [("a", 0, 2)]
        assert isinstance(plan.field(SAMPLING_MASK), RespRows) and isinstance(plan.field(INDEXER_TOPK), SeqRows)
        assert [r.key for r in plan.owned_refs] == ["a"]

    def test_unrequested_fields_are_none_and_requested_ones_must_be_stored(self):
        plan = response("a", prompt=3, output=2, requested={SAMPLING_MASK})
        assert plan.field(ROUTED_EXPERTS) is None and plan.field(INDEXER_TOPK) is None
        with pytest.raises(StorePlanError, match="asked for routed_experts"):
            StorePlan.from_response(ref("a", mask_rows=2), requested=ALL, num_tokens=5, response_length=2)

    def test_a_no_output_abort_gets_an_empty_mask_without_a_ref(self):
        plan = StorePlan.from_response(
            None, requested={SAMPLING_MASK}, num_tokens=3, response_length=0, no_output_abort=True
        )
        assert plan.field(SAMPLING_MASK) == RespRows() and plan.owned_refs == ()
        with pytest.raises(StorePlanError):
            StorePlan.from_response(None, requested={SAMPLING_MASK}, num_tokens=3, response_length=0)

    def test_row_counts_dtypes_and_support_logprobs_are_checked(self):
        with pytest.raises(StorePlanError, match="has 4 rows; expected 5"):
            StorePlan.from_response(ref("a", seq_rows=4), requested={ROUTED_EXPERTS}, num_tokens=6, response_length=1)
        with pytest.raises(StorePlanError, match="dtype int16"):
            StorePlan.from_response(
                ref("a", seq_rows=4, r3="int16"), requested={ROUTED_EXPERTS}, num_tokens=5, response_length=1
            )
        with pytest.raises(StorePlanError, match="support log-probs have 5 values for 6"):
            StorePlan.from_response(
                ref("a", mask_rows=2, support=True, logprob_entries=5),
                requested={SAMPLING_MASK},
                num_tokens=5,
                response_length=2,
            )

    def test_legacy_stop_edge_row_is_dropped(self):
        plan = StorePlan.from_response(
            ref("a", seq_rows=5), requested={ROUTED_EXPERTS}, num_tokens=5, response_length=1, allow_extra_seq_row=True
        )
        assert segs(plan, ROUTED_EXPERTS) == [("a", 0, 4)]
        with pytest.raises(StorePlanError, match="has 5 rows; expected 4"):
            StorePlan.from_response(ref("a", seq_rows=5), requested={ROUTED_EXPERTS}, num_tokens=5, response_length=1)


class TestEdits:
    def test_next_response_replaces_forward_rows_and_appends_mask_rows(self):
        first = response("a", prompt=3, output=2)
        second = response("b", prompt=5, output=3)
        plan = first.append_response(second, response_length=2)
        assert segs(plan, ROUTED_EXPERTS) == [("b", 0, 7)]
        assert segs(plan, SAMPLING_MASK) == [("a", 0, 2), ("b", 0, 3)]
        assert [r.key for r in plan.owned_refs] == ["a", "b"]
        with pytest.raises(StorePlanError, match="not aligned"):
            first.append_response(second, response_length=3)

    def test_a_tool_observation_adds_forced_mask_rows_only(self):
        plan = response("a", prompt=3, output=2).insert_tokens(2, response_length=2)
        assert segs(plan, SAMPLING_MASK) == [("a", 0, 2), (None, 0, 2)]
        assert segs(plan, ROUTED_EXPERTS) == [("a", 0, 4)]

    def test_session_merge_puts_forced_rows_between_turns_and_keeps_the_later_forward(self):
        a = response("a", prompt=3, output=2)
        b = response("b", prompt=7, output=2)
        plan = a.merge_turn(b, observation_tokens=2, response_length=2)
        assert segs(plan, SAMPLING_MASK) == [("a", 0, 2), (None, 0, 2), ("b", 0, 2)]
        assert segs(plan, ROUTED_EXPERTS) == [("b", 0, 8)]
        plan.validate(num_tokens=9, response_length=6)

    def test_a_forward_never_shrinks(self):
        long = response("a", prompt=6, output=2)
        short = response("b", prompt=3, output=1)
        with pytest.raises(StorePlanError, match="fewer than"):
            long.with_field(SAMPLING_MASK, None).append_response(
                short.with_field(SAMPLING_MASK, None), response_length=2
            )

    def test_strip_tail_cuts_across_segments_and_keeps_every_ref(self):
        plan = response("a", prompt=3, output=2).insert_tokens(2, response_length=2)
        plan = plan.append_response(response("b", prompt=7, output=2), response_length=4)
        stripped = plan.strip_tail(3)
        assert segs(stripped, SAMPLING_MASK) == [("a", 0, 2), (None, 0, 1)]
        assert segs(stripped, ROUTED_EXPERTS) == [("b", 0, 5)]
        assert [r.key for r in stripped.owned_refs] == ["a", "b"]

    def test_strict_none_rule(self):
        with_mask = response("a", prompt=3, output=2, requested={SAMPLING_MASK})
        with_r3 = response("b", prompt=5, output=2, requested={ROUTED_EXPERTS, SAMPLING_MASK})
        with pytest.raises(StorePlanError, match="with and without routed_experts"):
            with_mask.append_response(with_r3, response_length=2)
        assert combine_plans(None, None, lambda x, y: x) is None
        with pytest.raises(StorePlanError, match="one without"):
            combine_plans(with_mask, None, lambda x, y: x)


class TestValidate:
    def test_rows_must_match_the_sample(self):
        plan = response("a", prompt=3, output=2)
        plan.validate(num_tokens=5, response_length=2)
        with pytest.raises(StorePlanError, match="sampling_mask has 2 rows"):
            plan.validate(num_tokens=5, response_length=3)

    def test_segments_use_owned_objects_only(self):
        plan = response("a", prompt=3, output=2)
        foreign = plan.with_field(SAMPLING_MASK, RespRows((Seg(ref("x", mask_rows=2), 0, 2),)))
        with pytest.raises(StorePlanError, match="does not own"):
            foreign.validate(num_tokens=5, response_length=2)

    def test_sample_validate_checks_the_plan(self):
        sample = Sample(
            tokens=[1, 2, 3, 4, 5], response_length=2, rollout_store_plan=response("a", prompt=3, output=2)
        )
        sample.validate()
        sample.response_length = 1
        with pytest.raises(StorePlanError):
            sample.validate()


class TestTable:
    def test_round_trip_keeps_fields_forced_rows_empty_fields_and_unused_refs(self):
        plan = response("a", prompt=3, output=2).insert_tokens(2, response_length=2)
        plan = plan.append_response(response("b", prompt=7, output=2), response_length=4).strip_tail(2)
        plan = plan.with_field(INDEXER_TOPK, None).with_field(SAMPLING_MASK, plan.field(SAMPLING_MASK).prefix(0))

        table, refs = plan.to_table()

        assert table.dtype == np.int32 and table.shape[1] == 4
        assert [r.key for r in refs] == ["a", "b"]
        assert StorePlan.from_table(table, refs) == plan

    def test_out_of_range_rows_are_rejected(self):
        with pytest.raises(StorePlanError, match="out of range"):
            StorePlan.from_table(np.asarray([[0, 3, 0, 1]], dtype=np.int32), [ref("a")])


def test_dropped_samples_register_refs_once_without_the_kept_ones():
    take_dropped_refs()
    kept = Sample(rollout_store_plan=response("a", prompt=3, output=2))
    dropped = Sample(rollout_store_plan=response("b", prompt=3, output=2).adopt_refs([ref("c")]))
    register_dropped_samples([[kept, dropped], [Sample()]], [kept])
    register_dropped_samples([dropped])
    assert sorted(r.key for r in take_dropped_refs()) == ["b", "c"]
    assert take_dropped_refs() == []
