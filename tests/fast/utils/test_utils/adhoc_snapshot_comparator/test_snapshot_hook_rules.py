import copy

import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences, _wrap_hooks
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotHookWrap
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_hook_rules import _CONFIG_FIELDS, _scope_hooks


class TestSnapshotHookRules:
    @pytest.mark.parametrize("path,fields", _CONFIG_FIELDS.items())
    def test_declarations_match_the_live_hook_protocol(self, path: str, fields: tuple[str, ...] | None) -> None:
        """Every approved wrapper uses the real hook's declared configuration fields."""
        from miles.utils.args.custom_function import _compute_config_class
        from miles.utils.function_registry import load_function

        cls = _compute_config_class(load_function(path), path=path)
        assert (None if cls is None else set(cls.model_fields)) == (None if fields is None else set(fields))

    def test_shared_hook_fields_are_copied_without_changing_types(self) -> None:
        """Two hooks preserve the same original input before its flat ownership disappears."""
        path = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"
        args = {
            "rollout_function_path": path,
            "eval_function_path": path,
            "custom_async_data_buffer_path_per_model": False,
        }
        expected = {"case": {"config": copy.deepcopy(args)}}
        rules = _scope_hooks(args=args, scope=["case", "config"])
        _wrap_hooks(value=expected, rules=rules)
        actual = {
            "case": {
                "config": {
                    name: {"path": path, "config": {"custom_async_data_buffer_path_per_model": False}}
                    for name in ("rollout_function_path", "eval_function_path")
                }
            }
        }
        assert not _differences(before=expected, after=actual, path=[])
        actual["case"]["config"]["eval_function_path"]["config"]["custom_async_data_buffer_path_per_model"] = 0
        assert len(_differences(before=expected, after=actual, path=[])) == 1

    def test_missing_source_argument_is_not_filled_with_a_hook_default(self) -> None:
        """A wrapper cannot manufacture an argument absent from the captured before input."""
        path = "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn"
        assert _scope_hooks(args={"rollout_function_path": path}, scope=["case"]) == []
        value = {"case": {"rollout_function_path": path}}
        rule = SnapshotHookWrap(
            path=["case", "rollout_function_path"],
            function=path,
            config_fields=["custom_async_data_buffer_path_per_model"],
            op="test",
            reason="test",
        )
        with pytest.raises(AssertionError, match="Missing hook argument"):
            _wrap_hooks(value=value, rules=[rule])
