import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import app
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_audit import _write_audit
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotComparisonReport
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


class TestComparisonBinding:
    @pytest.mark.parametrize("changed", ["before", "after", "rules", "implementation", "legacy", "view", "report"])
    def test_reused_comparison_rejects_changed_inputs(self, tmp_path: Path, changed: str) -> None:
        """A saved comparison cannot acquire a new inventory or implementation identity."""
        before, after = tmp_path / "before", tmp_path / "after"
        for directory in (before, after):
            directory.mkdir()
            (directory / "case.yaml").write_text("value: 1\n")
        rules, output = tmp_path / "rules.json", tmp_path / "comparison.json"
        rules.write_text('{"moves": [], "approved_differences": []}')
        result = CliRunner().invoke(
            app, ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)]
        )
        assert result.exit_code == 0, result.exception
        saved = json.loads(output.read_text())
        actual = {"case.yaml": {"value": 1}}
        if changed in {"before", "after"}:
            (tmp_path / changed / "case.yaml").write_text("value: 2\n")
        elif changed == "rules":
            rules.write_text('{"moves": [], "approved_differences": [], "list_sorts": []}')
        elif changed == "implementation":
            saved["identity"]["implementation"]["miles_python_sha256"] = "0" * 64
            output.write_text(json.dumps(saved))
        elif changed == "legacy":
            del saved["identity"]
            output.write_text(json.dumps(saved))
        elif changed == "view":
            actual["case.yaml"]["value"] = 2
        report = SnapshotComparisonReport.model_validate(saved)
        if changed == "report":
            output.write_text(json.dumps({**saved, "verified_move_count": 1}))
        with pytest.raises(AssertionError):
            _write_audit(report=report, comparison_path=output, output=tmp_path / "audit.md", actual=actual)
        assert not (tmp_path / "audit.md").exists()


@pytest.mark.parametrize("variant", [False, True])
def test_rules_change_during_expansion_cannot_bind_old_rules_to_new_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: bool
) -> None:
    """The recorded rules hash must identify the exact bytes parsed before expansion."""
    from miles.utils.test_utils.adhoc_snapshot_comparator import snapshot_comparison, snapshot_variant_evidence

    module = snapshot_variant_evidence if variant else snapshot_comparison
    before, after = tmp_path / "before", tmp_path / "after"
    for directory in (before, after):
        directory.mkdir()
        (directory / "case.yaml").write_text("value: 1\n")
    rules, output = tmp_path / "rules.json", tmp_path / "comparison.json"
    rules.write_text('{"moves": [], "approved_differences": []}')
    expand = module.expand_snapshot_directory

    def change_rules(*, directory: Path) -> dict:
        result = expand(directory=directory)
        rules.write_text('{"moves": [], "approved_differences": [], "list_sorts": []}')
        return result

    monkeypatch.setattr(module, "expand_snapshot_directory", change_rules)
    arguments = ["--before", str(before), "--after", str(after), "--rules", str(rules), "--output", str(output)]
    if variant:
        arguments += ["--raw-directory", str(tmp_path)]
    result = CliRunner().invoke(module.app, arguments)
    assert isinstance(result.exception, AssertionError)
    assert "Rules changed during expansion" in str(result.exception)
    assert not output.exists()
