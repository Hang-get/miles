import json
from dataclasses import dataclass, field

from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference


@dataclass
class _ScopeEvidence:
    actual: dict[str, JsonValue]
    fields: dict[str, frozenset[str]]
    backend_fields: frozenset[str] = frozenset()
    witnesses: dict[tuple[str, str, str, str], list[str] | None] = field(default_factory=dict)

    @classmethod
    def load(cls, *, actual: dict[str, JsonValue], backend_fields: frozenset[str] = frozenset()) -> "_ScopeEvidence":
        from miles.utils.args.runtime import (
            AllConfig,
            InferenceControllerConfig,
            MultiLoraConfig,
            OrchestratorConfig,
            RolloutConfig,
            TrainerConfig,
        )

        return cls(
            actual=actual,
            backend_fields=backend_fields,
            fields={
                "all": frozenset(AllConfig.model_fields),
                "orchestrator": frozenset(OrchestratorConfig.model_fields),
                "rollout": frozenset(RolloutConfig.model_fields),
                "inference": frozenset(InferenceControllerConfig.model_fields),
                "trainer": frozenset(TrainerConfig.model_fields),
                "multi_lora": frozenset(MultiLoraConfig.model_fields),
            },
        )

    def explain(self, *, difference: SnapshotDifference) -> dict[str, JsonValue] | None:
        path = difference.path
        endpoint_reset = (
            path[-1] in {"sglang_model_routers", "sglang_router_ip", "sglang_router_port", "session_server_instances"}
            and difference.before.present
            and difference.before.value is not None
            and difference.after.present
            and difference.after.value is None
        )
        if not (
            path[0].startswith("runtime_config/")
            and len(path) == 6
            and path[-2] == "args"
            and difference.before.present
            and (not difference.after.present or endpoint_reset)
        ):
            return None
        process = path[1].rsplit("/", 1)[-1]
        role_by_process = {
            "rollout_executor": "rollout",
            "inference_controller": "inference",
            "multi_lora_controller": "multi_lora",
            "main": "orchestrator",
            "worker_manager": "orchestrator",
        }
        if process in role_by_process:
            role = role_by_process[process]
        else:
            case = self.actual[path[0]]
            assert isinstance(case, dict) and isinstance(captured := case[path[1]], dict)
            assert captured["ranks"] or process.startswith("trainer_controller_"), process
            role = "trainer"
        name = path[-1]
        if name in self.fields[role] and not endpoint_reset:
            return None
        if name not in self.fields["all"] and name not in self.backend_fields:
            return None
        witness = self._find_witness(
            case=path[0], generation=path[1].rsplit("/", 1)[0], name=name, value=difference.before.value
        )
        if witness is None:
            generation = path[1].rsplit("/", 1)[0]
            if name in {"cp_comm_type", "padded_vocab_size"} and difference.before.value is None:
                for selected_backend in ("fsdp", "torchtitan"):
                    backend = self._find_witness(
                        case=path[0], generation=generation, name="train_backend", value=selected_backend
                    )
                    if backend is not None:
                        return {
                            "rule": "scope/inactive-megatron-shim",
                            "verdict": "same-generation-inactive-backend-value-found",
                            "field": name,
                            "witness": backend,
                            "source": "miles/backends/torchtitan_utils/arguments.py:TorchtitanArgs; miles/backends/fsdp_utils/arguments.py:FSDPArgs",
                            "reason": "The exact old null Megatron shim has no FSDP or TorchTitan Namespace consumer; a same-generation record selects that backend. This lookup does not verify its owner, policy or consumer stage. Non-null and active Megatron values are not covered.",
                        }
            if name in {"load", "finetune", "no_load_optim", "no_load_rng"}:
                debug = self._find_witness(case=path[0], generation=generation, name="debug_rollout_only", value=True)
                if debug is not None:
                    return {
                        "rule": "lifecycle/rollout-only-unused-checkpoint",
                        "verdict": "same-generation-rollout-only-value-found",
                        "field": name,
                        "witness": debug,
                        "source": "miles/backends/megatron_utils/actor.py:MegatronTrainRayActor.init; tests/e2e/precision/test_dsv4_bshd_thd_rollout_parity.py:_run_rollout",
                        "reason": "A same-generation record contains debug_rollout_only=True. This lookup does not prove that this trainer or its consuming stage uses that record; the checkpoint explanation remains conditional.",
                    }
            if name == "load" and difference.before.value == "/root/models/Kimi-K3-4layer-64experts-bf16_torch_dist":
                changed = self._find_witness(
                    case=path[0],
                    generation=generation,
                    name="load",
                    value="/root/models/Kimi-K3-4layer-64experts-bf16-kda-direct-v1_torch_dist",
                )
                if changed is not None:
                    return {
                        "rule": "behavior/kimi-layout-cache-owner",
                        "verdict": "same-generation-changed-value-found",
                        "field": name,
                        "witness": changed,
                        "source": "historical after137 43d2546a6f4cb5e9095f0f37a4e3f1eba1189ace:scripts/run_kimi_k3.py:_torch_dist_name; M48 op48-101; op48-225",
                        "reason": "The historical captured owner has the exact KDA-layout cache path. Op48-225 later restores the standard BF16 path; this witness establishes a recorded value, not ownership, consumption or current-source agreement. The strict difference remains.",
                    }
        # Scope omission is audited, never removed from the strict comparison.
        # Matching checks the actual captured value and declared role boundary,
        # not the owner, policy or consuming stage. Those proofs remain open.
        return {
            "rule": "scope/endpoint-state-not-copied" if endpoint_reset else "scope/declared-role-exclusion",
            "verdict": "same-generation-value-found" if witness else "same-generation-value-not-found",
            "owner_verified": False,
            "consumer_verified": False,
            "scope": role,
            "field": name,
            "source": "after137:miles/utils/args/runtime.py (component model inheritance); miles/ray/specs/train.py and miles/ray/specs/rollout.py (config_class)",
            "reason": (
                "This component no longer receives the endpoint copy. The witness only locates an equal typed value in the same generation; it does not establish an endpoint owner or consuming stage."
                if endpoint_reset
                else "The receiving component does not declare this field. The witness only locates an equal typed value in the same generation, possibly in another policy or stage. Ownership and consumer equivalence are not verified; the strict difference remains."
            ),
            "witness": witness,
        }

    def _find_witness(self, *, case: str, generation: str, name: str, value: JsonValue) -> list[str] | None:
        fingerprint = json.dumps(value, sort_keys=True)
        key = (case, generation, name, fingerprint)
        if key in self.witnesses:
            return self.witnesses[key]
        data = self.actual.get(case)
        if isinstance(data, dict):
            for process_name, process in data.items():
                if process_name.rsplit("/", 1)[0] != generation:
                    continue
                if not isinstance(process, dict) or not isinstance(stages := process.get("stages"), dict):
                    continue
                for stage_name, stage in stages.items():
                    if not isinstance(stage, dict) or not isinstance(args := stage.get("args"), dict):
                        continue
                    owners = [(args, [])]
                    for owner_name in ("backend", "raw_fsdp", "raw_torchtitan"):
                        if isinstance(owner := args.get(owner_name), dict):
                            owners.append((owner, [owner_name]))
                    if isinstance(raw := args.get("raw_megatron"), dict) and isinstance(
                        owner := raw.get("base_args"), dict
                    ):
                        owners.append((owner, ["raw_megatron", "base_args"]))
                    for owner, owner_path in owners:
                        if name in owner and json.dumps(owner[name], sort_keys=True) == fingerprint:
                            found = [case, process_name, "stages", stage_name, "args", *owner_path, name]
                            self.witnesses[key] = found
                            return found
        self.witnesses[key] = None
        return None
