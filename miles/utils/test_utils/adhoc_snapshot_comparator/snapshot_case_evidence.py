from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _lookup
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference


def _case_evidence(*, difference: SnapshotDifference, actual: dict[str, JsonValue]) -> dict[str, JsonValue] | None:
    path = difference.path
    if path == ["parsed_config/fsdp_reject_default_ft.yaml"] and not difference.before.present:
        expected = {
            "error": {
                "message": "weight_update FT requires Megatron, --update-weight-transfer-mode p2p, and no --colocate",
                "type": "AssertionError",
            }
        }
        if difference.after.present and difference.after.value == expected:
            return {
                "rule": "case/fsdp-default-ft-negative-contract",
                "verdict": "intentional-negative-contract-coverage",
                "source": "miles/utils/arguments.py:_resolve_ft_components; tests/fast/argument_snapshot/parse_outputs/result_scenarios.py",
                "human_sot": "user-written/2026-07-15-impl-M48.md#op48-217",
                "reason": "The accepted upstream default now includes weight_update, which is Megatron-only. This new negative case retains the original default input; the two positive FSDP fixtures explicitly select rollout. The before success fixture cannot be substituted for this rejection capture.",
            }
    if (
        path[0] in {"parsed_config/megatron_rollout_ft.yaml", "parsed_config/megatron_ft_explicit_api.yaml"}
        and path[-1] == "ft_components"
        and difference.before.present
        and difference.after.present
        and difference.before.value == ["rollout"]
        and difference.after.value == ["rollout", "weight_update"]
    ):
        return {
            "rule": "case/upstream-default-ft-components",
            "verdict": "upstream-default-behavior-change",
            "source": "miles/utils/arguments.py:_FT_DEFAULT_COMPONENTS; miles/backends/training_utils/weight_update/rollout_cell_updater.py",
            "human_sot": "user-written/2026-07-15-impl-M48.md#op48-217",
            "reason": "These two unchanged Megatron fixture inputs omit ft_components. Upstream now enables weight_update FT as well as rollout; the consumer explicitly branches on that membership. This is an accepted main-version behavior boundary, not a mechanical rename or inferred equality.",
        }
    if path[0].startswith("parsed_config/"):
        scope = path[:2]
    elif path[0].startswith("runtime_config/") and len(path) >= 6:
        scope = path[:5]
    else:
        return None
    observed = _lookup(value=actual, path=scope)
    if not observed.present or not isinstance(args := observed.value, dict):
        return None
    field = path[-1]
    evidence = None
    if (
        path[0].startswith("runtime_config/")
        and len(path) == 7
        and path[2] == "stages"
        and path[3].startswith("checkpoint_load-")
        and path[4:] == ["args", "backend", "ckpt_step"]
        and not difference.before.present
        and difference.after.present
        and difference.after.value is None
        and isinstance(backend := args.get("backend"), dict)
        and backend.get("backend_name") == "fsdp"
        and args.get("train_backend", "fsdp") == "fsdp"
    ):
        evidence = {
            "rule": "case/fsdp-checkpoint-step-default",
            "verdict": "consumer-default-preserved",
            "source": "before:miles/backends/fsdp_utils/checkpoint.py:load; after:miles/backends/fsdp_utils/checkpoint.py:load",
            "witness": {"backend_name": "fsdp", "stage": path[3], "ckpt_step": None},
            "reason": "At the captured FSDP checkpoint load boundary, the old getattr(actor.args, 'ckpt_step', None) and the new actor.args.backend.ckpt_step both produce None and select the tracker branch. This covers only missing-to-null at this boundary, not a Megatron request/apply or a non-null checkpoint step.",
        }
    if (
        field == "critic_load"
        and difference.before.present
        and isinstance(difference.before.value, str)
        and difference.after.present
        and difference.after.value is None
        and args.get("train_backend") == "megatron"
        and args.get("use_critic") is False
    ):
        evidence = {
            "rule": "case/inactive-single-policy-critic-default",
            "verdict": "disabled-critic-consumer-verified",
            "source": "1a02469079139c363188c54ace020388821bdbab:miles/utils/arguments.py:miles_validate_args; miles/utils/args/trainer_utils.py:compute_trainer_ckpt_load_config; miles/backends/megatron_utils/megatron_config.py:_compute_trainers",
            "witness": {"train_backend": "megatron", "use_critic": False, "critic_load": None},
            "reason": "The recorded run disables critic creation. Op48-247 preserves an unset single-policy critic source until request resolution, rather than persisting the actor default. The exact old path-to-null difference is real, but no critic checkpoint consumer exists in this captured run. Active critics and explicit non-null new paths are not covered.",
        }
    if difference.before.present and not difference.after.present:
        if (
            field in {"cp_comm_type", "padded_vocab_size"}
            and difference.before.value is None
            and args.get("train_backend") == "fsdp"
        ):
            evidence = {
                "rule": "case/inactive-megatron-shim",
                "verdict": "disabled-consumer-verified",
                "source": "miles/backends/fsdp_utils/arguments.py; miles/utils/args/configs/custom_megatron_plugins.py",
                "witness": {"train_backend": args["train_backend"], "removed_value": None},
                "reason": "This is exactly a null Megatron-only compatibility declaration in an FSDP case; the complete consumer inventory has no FSDP read of these Namespace fields.",
            }
    if not difference.before.present and difference.after.present:
        tinker_defaults = {
            "tinker_base_model": None,
            "tinker_checkpoint_root": None,
            "tinker_lora_groups": None,
            "tinker_server_host": "0.0.0.0",
            "tinker_server_port": 10613,
        }
        if field in tinker_defaults and difference.after.value == tinker_defaults[field]:
            if path[0].startswith("parsed_config/") or path[1].endswith("/worker_manager"):
                evidence = {
                    "rule": "case/gateway-declarations-outside-gateway",
                    "verdict": "disabled-consumer-verified",
                    "source": "miles/utils/args/configs/tinker.py; serve_tinker.py; miles/tinker/arguments.py",
                    "witness": {"scope": scope, "declared_value": difference.after.value},
                    "reason": "These five fields are consumed only by the serve_tinker gateway. This captured scope is parser-only or WorkerManager, which does not start that gateway; values exactly match the added declarations.",
                }
        if field in {"hf_lora_targets", "lora_adapter_targets"} and difference.after.value is None:
            if (
                type(args.get("lora_rank")) is int
                and args["lora_rank"] == 0
                and "lora_adapter_path" in args
                and args["lora_adapter_path"] is None
            ):
                evidence = {
                    "rule": "case/disabled-lora-targets",
                    "verdict": "disabled-consumer-verified",
                    "source": "miles/utils/lora/utils.py:is_lora_enabled; miles/backends/megatron_utils/actor.py",
                    "witness": {"lora_rank": args["lora_rank"], "lora_adapter_path": args["lora_adapter_path"]},
                    "reason": "Both captured inputs disable the unchanged LoRA consumer predicate; the newly declared null targets are not consumed as an active adapter configuration.",
                }
        elif field in {"raw_fsdp", "raw_torchtitan"} and difference.after.value is None:
            if "train_backend" in args and args["train_backend"] != field.removeprefix("raw_"):
                evidence = {
                    "rule": "case/inactive-backend-null",
                    "verdict": "disabled-consumer-verified",
                    "source": "miles/utils/args/trainer_utils.py:compute_trainer_config",
                    "witness": {"train_backend": args["train_backend"]},
                    "reason": "The captured backend selects a different branch; compute_trainer_config requires the active backend to be non-null and never reads this inactive declaration.",
                }
        elif field == "backend_name" and "train_backend" in args and difference.after.value == args["train_backend"]:
            evidence = {
                "rule": "case/backend-tag",
                "verdict": "identity-verified",
                "source": "miles/utils/args/trainer_utils.py:compute_trainer_config",
                "witness": {"train_backend": args["train_backend"]},
                "reason": "The new discriminator exactly identifies the already selected backend; no implementation is switched.",
            }
    if evidence is not None:
        evidence["scope"] = scope
    return evidence
