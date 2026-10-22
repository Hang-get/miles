import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_failure_recovery import _recover_actual_blocks


def test_traceback_recovery_preserves_yaml_indentation_and_rejects_conflicts() -> None:
    """Recover printed YAML exactly without accepting contradictory or truncated copies."""
    block = "E           --- /root/miles/tests/snapshots/parsed_config/case.yaml\nE           --- BEGIN ACTUAL case.yaml ---\nE           config:\nE             value: false\nE           --- END ACTUAL ---\n"
    assert _recover_actual_blocks(text=block + block) == {"parsed_config/case.yaml": "config:\n  value: false\n"}
    with pytest.raises(AssertionError, match="Conflicting actual outputs"):
        _recover_actual_blocks(text=block + block.replace("value: false", "value: 0"))
    with pytest.raises(AssertionError, match="Truncated actual YAML block"):
        _recover_actual_blocks(text=block.replace("E           --- END ACTUAL ---\n", ""))
