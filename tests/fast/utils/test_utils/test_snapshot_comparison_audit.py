import json
from pathlib import Path

import pytest
from pydantic import JsonValue

from miles.utils.test_utils.snapshot_comparison_audit import _category, _write_audit
from miles.utils.test_utils.snapshot_comparison_models import (
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
