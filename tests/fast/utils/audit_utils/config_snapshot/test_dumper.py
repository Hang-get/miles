from argparse import Namespace
from pathlib import Path

import pytest

from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR


class TestCaptureState:
    @pytest.mark.parametrize("mode", ["not_ci", "disabled", "no_directory"])
    def test_reconfiguring_capture_off_does_not_leak_the_previous_process_state(
        self, capture_args: Namespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
    ) -> None:
        """A later disabled process cannot append records under an earlier process's identity."""
        if mode == "not_ci":
            capture_args.ci_test = False
        elif mode == "disabled":
            capture_args.ci_disable_config_snapshot = True
        else:
            monkeypatch.delenv(SNAPSHOT_RECORD_DIR_ENV_VAR)
        ConfigSnapshotDumper.configure(args=capture_args, source=SimpleProcessIdentity(component="rollout_executor"))

        ConfigSnapshotDumper.dump(stage="later_stage", config={"value": 9})

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert [record.point.stage for record in records] == ["process_config"]

    def test_stage_counters_are_independent_and_invalid_stages_consume_no_indices(
        self, capture_args: Namespace, tmp_path: Path
    ) -> None:
        """Interleaved stages retain consecutive per-stage occurrence numbers."""
        ConfigSnapshotDumper.dump(stage="train_first_step", config={})
        with pytest.raises(ValueError, match="Invalid snapshot stage"):
            ConfigSnapshotDumper.dump(stage="../invalid", config={})
        ConfigSnapshotDumper.dump(stage="checkpoint_load", config={})
        ConfigSnapshotDumper.dump(stage="train_first_step", config={})

        records = ConfigSnapshotStorage(directory=tmp_path).read()
        assert sorted((record.point.stage, record.point.index) for record in records) == [
            ("checkpoint_load", 0),
            ("process_config", 0),
            ("train_first_step", 0),
            ("train_first_step", 1),
        ]

    def test_enabled_capture_requires_a_name(self, capture_args: Namespace) -> None:
        """Unnamed captures must fail before producing unidentifiable records."""
        capture_args.config_snapshot_name = None
        with pytest.raises(ValueError, match="Snapshot name is required"):
            ConfigSnapshotDumper.configure(args=capture_args, source=SimpleProcessIdentity(component="main"))


class TestCaptureMembership:
    def test_late_binding_preserves_constructor_record_and_rejects_rebinding(
        self, capture_args: Namespace, tmp_path: Path
    ) -> None:
        """Learning the worker incarnation never rewrites the already captured constructor config."""
        from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotMembership
        from miles.utils.audit_utils.process_identity import TrainProcessIdentity

        ConfigSnapshotDumper.configure(
            args=capture_args, source=TrainProcessIdentity(component="actor", cell_index=0, rank_within_cell=0)
        )
        ConfigSnapshotDumper.dump(stage="process_config", config={"args": {"value": [1, False, None]}})
        original = {path: path.read_bytes() for path in tmp_path.glob("*.json")}
        membership = ConfigSnapshotMembership(
            cell_id="trainer-engine-actor-00000", workers_hash="actual-worker-hash", expected_ranks=[0]
        )
        capture_id = ConfigSnapshotDumper.bind(membership)
        assert ConfigSnapshotDumper.bind(membership) == capture_id
        with pytest.raises(AssertionError, match="cannot change incarnation"):
            ConfigSnapshotDumper.bind(membership.model_copy(update={"workers_hash": "replacement"}))
        ConfigSnapshotDumper.complete_initialization([capture_id])
        storage = ConfigSnapshotStorage(directory=tmp_path)
        assert len(storage.read_bindings()) == 1
        assert storage.read_bindings()[0].capture_id == capture_id
        assert storage.read_initializations()[0].capture_ids == [capture_id]
        assert {path: path.read_bytes() for path in tmp_path.glob("*.json")} == original

    def test_disabled_capture_writes_no_membership_or_completion(
        self, capture_args: Namespace, tmp_path: Path
    ) -> None:
        """Snapshot-off execution performs no metadata writes at the initialization boundary."""
        from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotMembership

        capture_args.ci_disable_config_snapshot = True
        ConfigSnapshotDumper.configure(args=capture_args, source=SimpleProcessIdentity(component="main"))
        assert not ConfigSnapshotDumper.enabled()
        assert (
            ConfigSnapshotDumper.bind(
                ConfigSnapshotMembership(cell_id="actor-00000", workers_hash="hash", expected_ranks=[0])
            )
            is None
        )
        ConfigSnapshotDumper.complete_initialization(["unused"])
        assert not (tmp_path / "bindings").exists()
        assert not (tmp_path / "initializations").exists()
