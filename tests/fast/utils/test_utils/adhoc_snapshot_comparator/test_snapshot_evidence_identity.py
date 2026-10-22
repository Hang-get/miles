from pathlib import Path

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity


class TestSnapshotIdentity:
    def test_identity_changes_for_values_paths_and_missing_files(self, tmp_path: Path) -> None:
        """Evidence cannot silently migrate to a changed snapshot inventory."""
        path = tmp_path / "case.yaml"
        path.write_text("value: false\n")
        original = _snapshot_identity(directory=tmp_path)["sha256"]
        path.write_text("value: 0\n")
        changed_value = _snapshot_identity(directory=tmp_path)["sha256"]
        assert changed_value != original
        path.rename(tmp_path / "renamed.yaml")
        changed_path = _snapshot_identity(directory=tmp_path)["sha256"]
        assert changed_path != changed_value
        path.write_text("value: 0\n")
        assert _snapshot_identity(directory=tmp_path)["sha256"] != changed_path
