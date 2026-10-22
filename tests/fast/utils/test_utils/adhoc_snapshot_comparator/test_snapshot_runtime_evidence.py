import json
from pathlib import Path

from typer.testing import CliRunner

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_runtime_evidence import _before_trainer_cells, app


def test_startup_count_uses_critic_gpu_shape_and_all_parallel_dimensions() -> None:
    """Replaying the old formula must not silently use actor resources for a critic."""
    args = {
        "indep_dp": True,
        "critic_num_nodes": 3,
        "critic_num_gpus_per_node": 8,
        "actor_num_nodes": 9,
        "actor_num_gpus_per_node": 8,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 3,
        "context_parallel_size": 2,
    }
    assert _before_trainer_cells(args=args, role="critic") == 2
    assert _before_trainer_cells(args=args, role="actor") == 6
    assert _before_trainer_cells(args={"indep_dp": False}, role="actor") == 1


class TestRuntimeAuditCoverage:
    def test_no_runtime_inputs_do_not_certify_startup_targets(self, tmp_path: Path) -> None:
        """An empty selected input set must not produce a successful audit exit code."""
        before, after = tmp_path / "before", tmp_path / "after"
        for directory in (before, after):
            directory.mkdir()
            (directory / "case.yaml").write_text("unrelated: true\n")
        output = tmp_path / "audit.json"
        result = CliRunner().invoke(app, ["--before", str(before), "--after", str(after), "--output", str(output)])
        assert result.exit_code == 1
        assert json.loads(output.read_text())["coverage_status"] == "no-inputs"

    def test_missing_captured_trainer_target_is_reported(self, tmp_path: Path) -> None:
        """Removing every target from an actual trainer cannot turn the audit into zero successful inputs."""
        before, after = tmp_path / "before", tmp_path / "after"
        for directory in (before, after):
            (directory / "runtime_config").mkdir(parents=True)
            process = {"ranks": [0], "base": {"args": {"trainer_id": "actor"}}, "diffs": {}}
            (directory / "runtime_config/case.yaml").write_text(
                json.dumps({"processes": {"run/actor_cell00000": process}})
            )
        output = tmp_path / "audit.json"
        result = CliRunner().invoke(app, ["--before", str(before), "--after", str(after), "--output", str(output)])
        assert result.exit_code == 1
        saved = json.loads(output.read_text())
        assert saved["counts"]["targets_not_captured"] == 1
        assert saved["coverage_status"] == "missing-trainer-targets"
