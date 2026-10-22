import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_audit import _category, _write_audit
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotComparisonReport,
    SnapshotDifference,
    SnapshotValue,
)


class TestSnapshotAudit:
    @pytest.mark.parametrize(
        "before,after,category",
        [
            (1, 1.0, "representation/numeric-type-change"),
            (False, 0, "value/type-change"),
            ([1, 2], [2, 1], "value/list-change"),
            ("module.fn", {"path": "module.fn", "config": {"new": True}}, "representation/hook-path-preserved"),
        ],
    )
    def test_classification_does_not_make_changed_values_equal(
        self, before: JsonValue, after: JsonValue, category: str
    ) -> None:
        """Diagnostic categories distinguish type changes without approving them."""
        difference = SnapshotDifference(
            path=["parsed_config/case.yaml", "config", "value"],
            before=SnapshotValue(present=True, value=before),
            after=SnapshotValue(present=True, value=after),
        )
        assert _category(difference=difference) == category

    def test_grouping_retains_all_typed_variants_and_reconciles_counts(self, tmp_path: Path) -> None:
        """Repeated stages collapse without dropping distinct typed evidence or missing cases."""
        differences = [
            SnapshotDifference(
                path=["runtime_config/case.yaml", "run/actor_cell00000", "stages", f"train-{index:04d}", "args", "x"],
                before=SnapshotValue(present=True, value=value),
                after=SnapshotValue(present=False, value=None),
            )
            for index, value in enumerate([False, 0, False])
        ]
        differences.append(
            SnapshotDifference(
                path=["runtime_config/missing.yaml"],
                before=SnapshotValue(present=False, value=None),
                after=SnapshotValue(present=True, value={}),
            )
        )
        report = SnapshotComparisonReport(
            before_directory="before",
            after_directory="after",
            moves=[],
            approved_differences=[],
            unexplained=differences,
            unused_approvals=[],
            list_sorts=[],
            sorting_differences=[],
        )
        output = tmp_path / "audit.md"
        _write_audit(report=report, output=output)
        audit = json.loads(output.with_suffix(".json").read_text())

        assert sum(audit["categories"].values()) == 4
        group = next(group for group in audit["groups"] if group["category"] == "presence/removed-field")
        assert group["count"] == 3
        assert sorted(variant["count"] for variant in group["variants"]) == [1, 2]
        assert len(report.unexplained) == 4 and report.approved_differences == []
        assert "Unresolved strict differences: 4" in output.read_text()

    def test_changed_text_case_is_not_reported_as_missing(self) -> None:
        """A changed command snapshot retains both cases and is a value difference."""
        difference = SnapshotDifference(
            path=["launch_scripts/case.txt"],
            before=SnapshotValue(present=True, value="old command"),
            after=SnapshotValue(present=True, value="new command"),
        )
        assert _category(difference=difference) == "value/scalar-change"

    @pytest.mark.parametrize(
        "value,verdict", [("zero", "consumer-default-preserved"), ("normal", "requires-specific-witness")]
    )
    def test_source_rationale_does_not_approve_an_unmatched_added_value(self, value: str, verdict: str) -> None:
        """A known consumer fallback cannot justify a different concrete value."""
        from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_catalog import _describe_difference

        difference = SnapshotDifference(
            path=["case", "lora_B_init_method"],
            before=SnapshotValue(present=False, value=None),
            after=SnapshotValue(present=True, value=value),
        )
        explanation = _describe_difference(difference=difference, category="presence/added-field")
        assert explanation["verdict"] == verdict

    def test_identical_values_keep_distinct_case_evidence(self, tmp_path: Path) -> None:
        """One case's owner witness cannot be reused for an equal value pair in another case."""
        actual = {
            f"runtime_config/{name}.yaml": {
                "run/main": {"stages": {"process_config-0000": {"args": {"partial_rollout": value}}}}
            }
            for name, value in (("matched", False), ("unmatched", True))
        }
        differences = [
            SnapshotDifference(
                path=[name, "run/inference_controller", "stages", "process_config-0000", "args", "partial_rollout"],
                before=SnapshotValue(present=True, value=False),
                after=SnapshotValue(present=False, value=None),
            )
            for name in actual
        ]
        report = SnapshotComparisonReport(
            before_directory="before",
            after_directory="after",
            moves=[],
            approved_differences=[],
            unexplained=differences,
            unused_approvals=[],
            list_sorts=[],
            sorting_differences=[],
        )
        output = tmp_path / "audit.md"
        _write_audit(report=report, output=output, actual=actual)
        audit = json.loads(output.with_suffix(".json").read_text())
        assert audit["evidence_counts"] == {"retained-owner-value-verified": 1, "owner-value-not-captured": 1}
        assert len(audit["groups"][0]["variants"]) == 2
        assert len(audit["evidence_records"]) == 2
        assert audit["evidence_records"][0]["witness"][0] == "runtime_config/matched.yaml"
        assert audit["evidence_records"][1]["witness"] is None


class TestVariantInventoryExplanation:
    def test_variant_key_does_not_assert_canonical_before_is_missing(self) -> None:
        """A new variant path requires explicit pairing rather than inventing missing execution."""
        from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_catalog import _describe_difference

        difference = SnapshotDifference(
            path=["runtime_config/variants/h200/wandb/tests/e2e/case.yaml"],
            before=SnapshotValue(present=False, value=None),
            after=SnapshotValue(present=True, value={}),
        )
        explanation = _describe_difference(difference=difference, category="coverage/missing-case")
        assert explanation["verdict"] == "variant-pairing-evidence-required"
        assert difference.before.present is False
