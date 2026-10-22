from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotDifference,
    SnapshotValue,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_parser_evidence import (
    _action_evidence,
    _actions_by_identity,
)


def test_positive_and_negative_actions_with_one_destination_are_both_preserved() -> None:
    """Parser inventory must not hide a removed negative flag behind its positive counterpart."""
    actions = [
        {"dest": "enabled", "option_strings": ["--enabled"], "const": True},
        {"dest": "enabled", "option_strings": ["--no-enabled"], "const": False},
    ]
    indexed = _actions_by_identity(actions=actions)
    assert len(indexed) == 2
    assert list(indexed.values()) == actions


def test_unknown_parser_action_change_is_not_marked_explained() -> None:
    """Unregistered action behavior retains an explicit unresolved verdict."""
    change = SnapshotDifference(
        path=['["unknown", ["--unknown"]]', "default"],
        before=SnapshotValue(present=True, value=False),
        after=SnapshotValue(present=True, value=0),
    )
    assert _action_evidence(difference=change)["verdict"] == "unverified-parser-change"


def test_numeric_converter_cannot_use_the_string_identity_rationale() -> None:
    """Only the exact implicit-to-string converter change has identity semantics."""
    change = SnapshotDifference(
        path=['["dsa_impl", ["--dsa-impl"]]', "type"],
        before=SnapshotValue(present=True, value=None),
        after=SnapshotValue(present=True, value={"$callable": "builtins.float"}),
    )
    assert _action_evidence(difference=change)["verdict"] == "unverified-parser-change"
