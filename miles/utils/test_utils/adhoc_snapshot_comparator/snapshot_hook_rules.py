from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotHookWrap

# M30 SoT allows explicit structural moves, never synthesized argument defaults.
# custom_function._resolve_custom_function_config preserves the path and selects
# declared hook fields; custom_view.compute_custom_function_config merges them
# back for consumers. function_registry.load_function unwraps the same path.
# The None below is the no-hook-config representation, not a missing user value.
_CONFIG_FIELDS: dict[str, tuple[str, ...] | None] = {
    "examples.infra_features.fully_async.external_eval_fn.ExternalSglangEvalFn": None,
    "examples.multi_policy.solver_verifier.generate": None,
    "examples.multi_policy.solver_verifier.split_eval_data_by_policy": None,
    "miles.ray.rollout.external_engine_provider.static_inference_engine_provider": None,
    "miles.ray.specs.inference.backend_inference_engine_provider": None,
    "miles.rollout.data_source.RolloutDataSource": None,
    "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn": None,
    "miles.rollout.sft_rollout.generate_rollout": None,
    "miles.rollout.sglang_rollout.generate_rollout": None,
    "miles.utils.test_utils.anthropic_session_verify_agent.run_agent": None,
    "miles.utils.test_utils.session_verify_agent.run_agent": None,
    "miles_plugins.models.inkling.model.inkling_model_provider": None,
    "miles_plugins.models.qwen3_8_next.model_provider.get_qwen3_8_next_model_provider": None,
    "tests.e2e.precision.test_dsv4_bshd_thd_rollout_parity.deterministic_index_reward": None,
    # FullyAsyncRolloutFn.add_arguments delegates to add_data_buffer_arguments.
    "miles.rollout.fully_async_rollout.FullyAsyncRolloutFn": ("custom_async_data_buffer_path_per_model",),
    # Both generate functions expose add_arguments; the Anthropic variant adds
    # exactly one field to session_verify_agent.generate's four legacy fields.
    "miles.utils.test_utils.session_verify_agent.generate": (
        "custom_agent_function_mode",
        "max_seq_len",
        "session_verify_cycles",
        "tool_call_failure_mode",
    ),
    "miles.utils.test_utils.anthropic_session_verify_agent.generate": (
        "custom_agent_function_mode",
        "max_seq_len",
        "session_verify_cycles",
        "tool_call_failure_mode",
        "anthropic_intermediate_system_expectation",
    ),
}
_HOOK_FIELDS = frozenset(
    {
        "custom_agent_function_path",
        "custom_eval_rollout_log_function_path",
        "custom_generate_function_path",
        "custom_inference_engine_provider_path",
        "custom_model_provider_path",
        "custom_rm_path",
        "data_source_path",
        "eval_function_path",
        "rollout_function_path",
    }
)


def _complete_hook_wraps(*, snapshots: dict[str, JsonValue]) -> list[SnapshotHookWrap]:
    result: list[SnapshotHookWrap] = []
    for case_name, case in snapshots.items():
        if not isinstance(case, dict):
            continue
        if case_name.startswith("parsed_config/") and isinstance(args := case.get("config"), dict):
            result.extend(_scope_hooks(args=args, scope=[case_name, "config"]))
        elif case_name.startswith("runtime_config/"):
            for process_name, process in case.items():
                assert isinstance(process, dict) and isinstance(stages := process["stages"], dict)
                for stage_name, stage in stages.items():
                    assert isinstance(stage, dict) and isinstance(args := stage["args"], dict)
                    result.extend(
                        _scope_hooks(args=args, scope=[case_name, process_name, "stages", stage_name, "args"])
                    )
    return result


def _scope_hooks(*, args: dict[str, JsonValue], scope: list[str]) -> list[SnapshotHookWrap]:
    result = []
    for name in sorted(_HOOK_FIELDS & args.keys()):
        path = args[name]
        if not isinstance(path, str) or path not in _CONFIG_FIELDS:
            continue
        fields = _CONFIG_FIELDS[path]
        if fields is not None and not all(field in args for field in fields):
            # A missing original input cannot be replaced with the new default.
            continue
        result.append(
            SnapshotHookWrap(
                path=[*scope, name],
                function=path,
                config_fields=None if fields is None else list(fields),
                op="M30 custom-function configuration ownership",
                reason="Fixed after137 custom_function.py selects existing hook fields; custom_view.py merges the same values for consumers; function_registry.py resolves the unchanged path",
            )
        )
    return result
