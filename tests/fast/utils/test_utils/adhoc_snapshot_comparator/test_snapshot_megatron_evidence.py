from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_megatron_evidence import _add_actor_indices


def test_actor_indices_preserve_original_actor_order_and_leave_critic_unindexed() -> None:
    """Derive indices from old actor order without altering original fields."""
    trainers = [{"role": "actor", "id": "a"}, {"role": "critic", "id": "c"}, {"role": "actor", "id": "b"}]
    result = _add_actor_indices(trainers=trainers)
    assert [row["actor_index"] for row in result] == [0, None, 1]
    assert [{key: value for key, value in row.items() if key != "actor_index"} for row in result] == trainers
    assert all("actor_index" not in row for row in trainers)
