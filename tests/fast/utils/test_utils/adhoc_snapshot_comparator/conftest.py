from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotComparisonReport
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _comparison_identity


@pytest.fixture
def bind_audit_report(tmp_path: Path) -> Callable:
    def bind(
        report: SnapshotComparisonReport, actual: dict[str, JsonValue] | None = None
    ) -> tuple[SnapshotComparisonReport, Path]:
        before, after, rules = tmp_path / "before", tmp_path / "after", tmp_path / "rules.json"
        for directory in (before, after):
            directory.mkdir()
            (directory / "case.txt").write_text("fixture")
        rules.write_text('{"moves": [], "approved_differences": []}')
        report = report.model_copy(
            update={
                "before_directory": str(before),
                "after_directory": str(after),
                "identity": _comparison_identity(before=before, after=after, rules=rules, actual=actual or {}),
            }
        )
        path = tmp_path / "comparison.json"
        path.write_text(report.model_dump_json())
        return report, path

    return bind
