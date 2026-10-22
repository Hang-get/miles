from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_variant_evidence import _variant_relations


class TestVariantRelations:
    def test_equal_canonical_does_not_prove_missing_variant_before(self) -> None:
        """An equal canonical capture cannot supply an unrecorded hardware variant."""
        canonical = "runtime_config/tests/e2e/example.yaml"
        variant = "runtime_config/variants/h200/wandb/tests/e2e/example.yaml"
        rows = _variant_relations(before={canonical: {}}, after={canonical: {}, variant: {}})
        assert rows[0]["canonical_before_present"] is True
        assert rows[0]["canonical_difference_count"] == 0
        assert rows[0]["pairing_verdict"] == "no-explicit-variant-before-key"

    def test_all_typed_values_and_rank_order_remain_different(self) -> None:
        """Variant reporting preserves type, list order and new process differences."""
        canonical = "runtime_config/tests/e2e/example.yaml"
        variant = "runtime_config/variants/h100/no-wandb/tests/e2e/example.yaml"
        rows = _variant_relations(
            before={},
            after={
                canonical: {"ranks": [0, 1], "flag": 1, "nullable": None},
                variant: {"ranks": [1, 0], "flag": True, "new_process": {}},
            },
        )
        assert rows[0]["canonical_difference_count"] == 4
        assert {row["path"][-1] for row in rows[0]["canonical_differences"]} == {
            "ranks",
            "flag",
            "nullable",
            "new_process",
        }
        assert rows[0]["canonical_before_present"] is False


class TestIncarnationAlignment:
    def test_unique_incarnation_preserves_every_observed_difference(self) -> None:
        """A unique owner suffix changes no ranks or config values during comparison."""
        from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_variant_evidence import _align_single_incarnations

        original = {"actor": {"ranks": [0, 1], "value": 1}}
        observed = {"actor/trainer-actor-00000/incarnation-0000": {"ranks": [0], "value": 2}}
        aligned, moves = _align_single_incarnations(original=original, observed=observed)
        assert aligned == {"actor/trainer-actor-00000/incarnation-0000": {"ranks": [0, 1], "value": 1}}
        assert len(moves) == 1
        assert original == {"actor": {"ranks": [0, 1], "value": 1}}

    def test_multiple_incarnations_are_not_assigned_an_arbitrary_old_owner(self) -> None:
        """Distinct worker incarnations cannot be collapsed to hide lifecycle changes."""
        from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_variant_evidence import _align_single_incarnations

        original = {"actor": {}}
        observed = {"actor/trainer-actor-00000/incarnation-0000": {}, "actor/trainer-actor-00000/incarnation-0001": {}}
        assert _align_single_incarnations(original=original, observed=observed) == (original, [])


def test_variant_keeps_transformed_before_separate_from_historical_after() -> None:
    """A refreshed after cannot overwrite the original comparison operand."""
    canonical = "runtime_config/tests/e2e/example.yaml"
    variant = "runtime_config/variants/h200/wandb/tests/e2e/example.yaml"
    rows = _variant_relations(before={canonical: {"value": 1}}, after={canonical: {"value": 2}, variant: {"value": 3}})
    assert rows[0]["canonical_differences"][0]["before"]["value"] == 2
    assert rows[0]["transformed_before_differences"][0]["before"]["value"] == 1
    assert rows[0]["transformed_before_differences"][0]["after"]["value"] == 3
