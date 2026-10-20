"""Trainer hook of the output store path: one read per (object, field), columns of the inline path, no deletes."""

from argparse import Namespace

import numpy as np
import pytest
import torch

from miles.backends.training_utils.data.store_columns import materialize_store_columns
from miles.utils.output_store import config as output_store_config
from miles.utils.output_store.materialize import StoreExecError
from miles.utils.output_store.plan import (
    INDEXER_TOPK,
    ROUTED_EXPERTS,
    SAMPLING_MASK,
    StorePlan,
    ref_from_builtins,
    ref_to_builtins,
)

from tests.fast.fixtures.output_store_fixtures import make_store, mask_arrays, put_response  # isort: skip


@pytest.fixture
def store(monkeypatch):
    store, backend = make_store()
    monkeypatch.setattr(output_store_config, "_READER", store)
    return store, backend


def _args(**overrides) -> Namespace:
    values = dict(
        output_store_local_buffer_size="1kb",
        rollout_sampling_logprobs_mode="selected",
        rollout_top_logprobs_num=0,
        loss_type="policy_loss",
        ci_test=False,
        rollout_indexer_topk_num_streams=None,
    )
    return Namespace(**(values | overrides))


def _shard(plans: list[StorePlan], tokens: list[list[int]], response_lengths: list[int], **columns) -> dict:
    tables = [plan.to_table() for plan in plans]
    return {
        "tokens": tokens,
        "response_lengths": response_lengths,
        "loss_masks": [[1] * n for n in response_lengths],
        "store_plan": [table for table, _ in tables],
        "store_refs": [[ref_to_builtins(ref) for ref in refs] for _, refs in tables],
        **columns,
    }


def _r3(rows: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(1, 64, size=(rows, 2, 4), dtype=np.int32)


def test_forward_rows_of_a_shared_object_are_read_once_and_never_deleted(store):
    client, backend = store
    prefix = ref_from_builtins(put_response(client, routed_experts=_r3(4, 1)))
    leaf = ref_from_builtins(put_response(client, routed_experts=_r3(6, 2)))
    full = StorePlan.from_response(leaf, requested={ROUTED_EXPERTS}, num_tokens=7, response_length=2)
    # A second sample planned on the shared object only, owning both objects.
    short = StorePlan.from_response(prefix, requested={ROUTED_EXPERTS}, num_tokens=5, response_length=1)
    short = short.adopt_refs([leaf])
    data = _shard([full, short], [[1] * 7, [1] * 5], [2, 1])

    materialize_store_columns(_args(), data)

    assert "store_plan" not in data and "store_refs" not in data
    np.testing.assert_array_equal(data["rollout_routed_experts"][0], _r3(6, 2))
    np.testing.assert_array_equal(data["rollout_routed_experts"][1], _r3(4, 1))
    assert data["rollout_routed_experts"][0].dtype == np.int32
    assert sorted(key for read in backend.reads for key, _ in read) == sorted(
        [f"{prefix.key}/routed_experts", f"{leaf.key}/routed_experts"]
    )
    assert len(backend.parts) == 2  # the rollout executor deletes, never the trainer


def test_reads_are_chunked_by_the_buffer_and_released_before_the_next_chunk(store):
    client, backend = store
    refs = [ref_from_builtins(put_response(client, routed_experts=_r3(8, i))) for i in range(3)]
    plans = [StorePlan.from_response(r, requested={ROUTED_EXPERTS}, num_tokens=9, response_length=1) for r in refs]
    one_object = 8 * 2 * 4 * 4
    backend.capacity_bytes = one_object

    data = _shard(plans, [[1] * 9] * 3, [1] * 3)
    materialize_store_columns(_args(output_store_local_buffer_size=str(one_object)), data)

    assert len(backend.reads) == 3 and backend.live_bytes == 0


def test_sampling_mask_columns_match_the_inline_layout_with_forced_rows(store):
    client, _ = store
    supports = [[5, 6], [7, 8, 9]]
    logprobs = [[-0.5, -1.0], [-0.1, -0.2, -0.3]]
    mask_ref = ref_from_builtins(put_response(client, **mask_arrays(supports, logprobs)))
    plan = StorePlan.from_response(mask_ref, requested={SAMPLING_MASK}, num_tokens=4, response_length=2)
    plan = plan.insert_tokens(1, response_length=2)
    # Response tokens: the sampled 6 and 8, then a forced 40.
    data = _shard([plan], [[1, 2, 6, 8, 40]], [3], rollout_log_probs=[[-1.0, -0.2, 0.0]])

    materialize_store_columns(_args(rollout_sampling_logprobs_mode="support", rollout_top_logprobs_num=3), data)

    assert data["rollout_sampling_mask_ids"][0].tolist() == [5, 6, 7, 8, 9, 40]
    assert data["rollout_sampling_mask_ids"][0].dtype == torch.int32
    assert data["rollout_sampling_mask_offsets"][0].tolist() == [0, 2, 5, 6]
    assert data["rollout_sampling_mask_offsets"][0].dtype == torch.int64
    assert data["rollout_topk_token_ids"][0].tolist() == [[5, 6, -1], [7, 8, 9], [-1, -1, -1]]
    np.testing.assert_array_equal(
        data["rollout_topk_log_probs"][0][1], np.asarray([-0.1, -0.2, -0.3], dtype=np.float32)
    )


def test_score_centering_compacts_support_candidates(store):
    client, _ = store
    mask_ref = ref_from_builtins(put_response(client, **mask_arrays([[5, 6]], [[-0.5, -1.0]])))
    plan = StorePlan.from_response(mask_ref, requested={SAMPLING_MASK}, num_tokens=3, response_length=1)
    plan = plan.insert_tokens(1, response_length=1)
    data = _shard([plan], [[1, 2, 6, 40]], [2], rollout_log_probs=[[-1.0, 0.0]])

    materialize_store_columns(
        _args(rollout_sampling_logprobs_mode="support", rollout_top_logprobs_num=2, loss_type="score_centering"), data
    )

    assert "rollout_topk_token_ids" not in data
    assert data["rollout_topk_lengths"][0].tolist() == [2, 0]
    assert data["rollout_topk_lengths"][0].dtype == torch.int32


def test_a_sampled_token_outside_its_support_fails(store):
    client, _ = store
    mask_ref = ref_from_builtins(put_response(client, **mask_arrays([[5, 6]])))
    plan = StorePlan.from_response(mask_ref, requested={SAMPLING_MASK}, num_tokens=3, response_length=1)
    with pytest.raises(StoreExecError, match="absent from its sampling support"):
        materialize_store_columns(_args(), _shard([plan], [[1, 2, 9]], [1]))


def test_all_zero_routed_experts_and_wrong_indexer_streams_fail(store):
    client, _ = store
    zero = ref_from_builtins(put_response(client, routed_experts=np.zeros((2, 2, 4), dtype=np.int32)))
    plan = StorePlan.from_response(zero, requested={ROUTED_EXPERTS}, num_tokens=3, response_length=1)
    with pytest.raises(StoreExecError, match="all zeros"):
        materialize_store_columns(_args(), _shard([plan], [[1, 2, 3]], [1]))

    indexer = ref_from_builtins(put_response(client, indexer_topk=_r3(2, 3)))
    plan = StorePlan.from_response(indexer, requested={INDEXER_TOPK}, num_tokens=3, response_length=1)
    with pytest.raises(ValueError, match="2 streams but the model has 4"):
        materialize_store_columns(_args(rollout_indexer_topk_num_streams=4), _shard([plan], [[1, 2, 3]], [1]))


def test_shards_without_store_columns_are_untouched():
    data = {"tokens": [[1, 2]], "rollout_routed_experts": [np.ones((1, 2, 4), dtype=np.int32)]}
    materialize_store_columns(_args(), data)
    assert list(data) == ["tokens", "rollout_routed_experts"]
