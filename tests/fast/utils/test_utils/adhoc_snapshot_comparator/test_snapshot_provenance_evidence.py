import json
from pathlib import Path

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotDifference,
    SnapshotValue,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import _provenance_evidence


def test_provenance_does_not_prove_a_different_digest_at_the_same_path() -> None:
    """A source witness must match the exact compared value, not merely the key name."""
    difference = SnapshotDifference(
        path=["runtime_config/case", "process", "stages", "process_config-0000", "args", "wandb_key"],
        before=SnapshotValue(present=True, value="redacted-sha256:original"),
        after=SnapshotValue(present=True, value="$WANDB_KEY_environment"),
    )
    row = difference.model_dump() | {"witness": {"record": "raw-record.json"}}
    provenance = {json.dumps(difference.path): row}
    assert _provenance_evidence(difference=difference, provenance=provenance) is not None
    changed = difference.model_copy(update={"before": SnapshotValue(present=True, value="redacted-sha256:changed")})
    assert _provenance_evidence(difference=changed, provenance=provenance) is None


def test_loading_provenance_rejects_a_different_registered_digest(tmp_path: Path) -> None:
    """A registry claim for another digest cannot certify the compared credential representation."""
    import pytest

    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import (
        _load_provenance,
        _normalizer_hash,
    )

    row = {
        "path": ["runtime_config/case", "process", "stages", "process_config-0000", "args", "wandb_key"],
        "before": {"present": True, "value": "redacted-sha256:aaaaaaaaaaaaaaaa"},
        "after": {"present": True, "value": "$WANDB_KEY_environment"},
        "witness": {
            "full_normalized_stage_matches": True,
            "registered_kind": "wandb_key",
            "registered_name": "environment",
            "registered_redacted_value": "redacted-sha256:bbbbbbbbbbbbbbbb",
            "sha256": "a" * 64,
            "point": {"stage": "process_config", "index": 0},
        },
    }
    inventory = tmp_path / "snapshots"
    inventory.mkdir()
    (inventory / "case.yaml").write_text("value: false\n")
    source = tmp_path / "provenance.json"
    source.write_text(
        json.dumps(
            {
                "normalizer_sha256": _normalizer_hash(),
                "after_inventory": _snapshot_identity(directory=inventory),
                "proof": [row],
            }
        )
    )
    with pytest.raises(AssertionError):
        _load_provenance(path=source, after_directory=inventory)


def test_loading_provenance_rejects_a_changed_capture_inventory(tmp_path: Path) -> None:
    """An old same-case witness cannot certify a newly collected snapshot."""
    import pytest

    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import (
        _load_provenance,
        _normalizer_hash,
    )

    inventory = tmp_path / "snapshots"
    inventory.mkdir()
    target = inventory / "case.yaml"
    target.write_text("value: false\n")
    source = tmp_path / "provenance.json"
    source.write_text(
        json.dumps(
            {
                "normalizer_sha256": _normalizer_hash(),
                "after_inventory": _snapshot_identity(directory=inventory),
                "proof": [],
            }
        )
    )
    assert _load_provenance(path=source, after_directory=inventory) == {}
    target.write_text("value: 0\n")
    with pytest.raises(AssertionError, match="Stale provenance inventory"):
        _load_provenance(path=source, after_directory=inventory)


def test_historical_attempt_is_not_assigned_a_hardware_variant() -> None:
    """Historical metadata stays canonical while a real variant retains its identity."""
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import _attempt_case

    old = {"test": "tests/e2e/example.py", "completed": True}
    assert _attempt_case(data=old) == ("runtime_config/tests/e2e/example.yaml", True)
    assert _attempt_case(data=old | {"variant": {"hardware": "h200", "wandb_available": False}}) == (
        "runtime_config/variants/h200/no-wandb/tests/e2e/example.yaml",
        True,
    )


def test_unknown_raw_schema_is_retained_without_a_provenance_claim(tmp_path: Path) -> None:
    """Unreadable historical records remain explicit gaps, never fabricated witnesses."""
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import _build_provenance

    before = tmp_path / "before"
    after = tmp_path / "after"
    for directory in (before, after):
        (directory / "parsed_config").mkdir(parents=True)
        (directory / "parsed_config/case.yaml").write_text("{}\n")
    raw = tmp_path / "raw"
    (raw / "records").mkdir(parents=True)
    (raw / "attempt.json").write_text(json.dumps({"test": "tests/e2e/example.py", "completed": True}))
    source = raw / "records/record.json"
    source.write_text(json.dumps({"allocated_endpoints": []}))
    output = tmp_path / "proof.json"
    _build_provenance(before=before, after=after, raw_directory=raw, output=output)
    report = json.loads(output.read_text())
    assert report["proof"] == []
    assert report["rejected_attempts"][0]["verdict"] == "historical-schema-replay-required"
    assert str(source) in report["rejected_attempts"][0]["raw_sha256"]
    assert "allocated_endpoints" in report["rejected_attempts"][0]["error"]
