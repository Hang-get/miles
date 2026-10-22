import argparse
import json

import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotDifference,
    SnapshotValue,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_dependency_evidence import (
    _CHANGES,
    _dependency_evidence,
)


class TestDependencyEvidence:
    @pytest.mark.parametrize("value,matched", [(False, True), (0, False), (True, False)])
    def test_only_exact_pinned_default_gets_a_rationale(self, value: bool | int, matched: bool) -> None:
        """A dependency rationale must not cover changed values or bool/int coercion."""
        difference = SnapshotDifference(
            path=["case", "args", "sglang", "base_args", "enable_watermark"],
            before=SnapshotValue(present=False, value=None),
            after=SnapshotValue(present=True, value=value),
        )
        assert (_dependency_evidence(difference=difference) is not None) == matched

    def test_unrelated_owner_does_not_borrow_dependency_rationale(self) -> None:
        """An equal field name in another component is not dependency evidence."""
        difference = SnapshotDifference(
            path=["case", "args", "enable_watermark"],
            before=SnapshotValue(present=False, value=None),
            after=SnapshotValue(present=True, value=False),
        )
        assert _dependency_evidence(difference=difference) is None

    def test_documented_defaults_match_the_live_dependency_parser(self) -> None:
        """Pinned explanations fail if the active dependency declarations drift."""
        from miles.backends.sglang_utils.sglang_config import _add_prefixed_primary_server_args

        parser = argparse.ArgumentParser(add_help=False)
        _add_prefixed_primary_server_args(parser)
        actions = {action.dest: action for action in parser._actions}
        for field, (_, _, present, value, _, _) in _CHANGES.items():
            name = "sglang_" + field
            assert (name in actions) == present, field
            if present:
                assert json.dumps(actions[name].default) == json.dumps(value), field
