import json
from dataclasses import dataclass

from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference

_BEFORE = "4a8ed9962cd9e878565d748a0dbdd196cab958da"
_AFTER = "5cb2887dab2b3eedb2a947cbded921b912b870c9"
_M30 = "user-written/2026-07-15-impl-M30.md"
_M48 = "user-written/2026-07-15-impl-M48.md"


@dataclass(frozen=True)
class _EvidenceRule:
    name: str
    verdict: str
    explanation: str
    sources: list[str]
    decision: str
    next_check: str


def _describe_difference(*, difference: SnapshotDifference, category: str) -> dict[str, JsonValue]:
    field = difference.path[-1]
    rule = _rule(field=field, category=category)
    if category == "coverage/missing-case" and difference.path[0].startswith("runtime_config/variants/"):
        rule = _EvidenceRule(
            "capture/explicit-hardware-variant-key",
            "variant-pairing-evidence-required",
            "The strict file inventory has no identical before key for this hardware/tracking variant. This does not establish that its canonical before capture is absent. snapshot_variant_evidence records the explicit canonical pair, unique process correspondences, raw converter replay and all typed differences; snapshot_variant_audit checks the actual fresh owners. No variant is silently aliased by the strict comparator.",
            [
                "miles/utils/test_utils/adhoc_snapshot_comparator/snapshot_variant_evidence.py",
                "miles/utils/test_utils/adhoc_snapshot_comparator/snapshot_variant_audit.py",
            ],
            _M30 + "#SoT: 如何处理snapshot test",
            "Require exact before raw recovery, actual hardware/tracking provenance and the complete separately retained variant comparison. Keep this strict inventory difference even after successful pairing.",
        )
    if difference.path[0].startswith("launch_scripts/py/scripts/amd/run_gpt_oss_20b_lora.py/"):
        rule = _rule(field=field, category="behavior/amd-launcher-api-fix")
    if difference.path[0].startswith("argument_schema/") and field != "add_argument":
        rule = _rule(field=field, category="schema/parser-contract")
    expected = {
        "lora_B_init_method": "zero",
        "offload": False,
        "hf_lora_targets": None,
        "lora_adapter_targets": None,
        "num_layers": None,
        "custom_agent_function_path": None,
        "rollout_indexer_topk_num_streams": None,
    }
    if field in expected and (
        difference.before.present
        or not difference.after.present
        or json.dumps(difference.after.value, sort_keys=True) != json.dumps(expected[field], sort_keys=True)
    ):
        rule = _rule(field="unmatched-value", category="unverified")
    if category == "representation/numeric-type-change":
        pairs = {"min_batch_collection_ratio": [1, 1.0], "tis_clip_low": [0, 0.0], "kl_coef": [0, 0.0]}
        pair = [difference.before.value, difference.after.value]
        if field not in pairs or json.dumps(pair) != json.dumps(pairs[field]):
            rule = _rule(field="unmatched-numeric-pair", category="unverified")
    return {
        "rule": rule.name,
        "verdict": rule.verdict,
        "explanation": rule.explanation,
        "historical_source_revisions": {"before": _BEFORE, "after": _AFTER},
        "source_validation": "Historical source rationale; current inventory identity and rerun evidence are reported separately.",
        "sources": rule.sources,
        "human_sot": rule.decision,
        "next_check": rule.next_check,
        # The surrounding variant includes the exact typed before/after values
        # and a concrete path. A rationale is never an approval or filter.
    }


def _rule(*, field: str, category: str) -> _EvidenceRule:
    if category == "behavior/amd-launcher-api-fix":
        return _EvidenceRule(
            "behavior/amd-command-backend",
            "documented-and-reproduced-bug-fix",
            "The script exists before122, but all prepare/train/full_train entries call removed module APIs and raise AttributeError before producing a successful recording. CPU replay reproduced all six test errors. Commit 6e610ad394 replaces these calls with args.create_backend(), fixing the old bug; the three after-only successful outputs are expected evidence of that fix.",
            [
                "scripts/amd/run_gpt_oss_20b_lora.py:_download_inputs",
                "scripts/amd/run_gpt_oss_20b_lora.py:_execute",
                "tests/manual/launch_scripts/test_py_launch_scripts.py",
            ],
            _M48 + "#op48-190",
            "Retain before-amd-001.log as the negative before result and verify the three after entries. Do not invent successful before command snapshots or classify the script as newly added.",
        )
    if category == "schema/parser-contract":
        return _EvidenceRule(
            "schema/parser-contract-inventory",
            "audited-cli-surface-change",
            "The lossless action inventory preserves (dest, option_strings), defaults, const, action class and ordering. The current paired parser inventory separately reports every action addition/removal and field change, including dependency-version defaults, action classes, option aliases and accepted choices; historical common-default equality does not apply to refreshed dependency captures. The refactor also removes the unused FSDP padded-vocab-size option, changes groups/help and three implicit-string-to-str declarations, and changes ordering. Legacy rollout/generate hooks now register their declared config where the old legacy branch skipped them, so snapshot_hook=23 is an intentional compatibility capability change rather than numerical drift.",
            [
                "miles/utils/args/schema.py",
                "miles/utils/args/custom_function.py:add_user_provided_function_arguments",
                "before:miles/utils/arguments.py:get_miles_extra_args_provider",
                "tests/fast/argument_snapshot/argparser/scenarios.py",
            ],
            _M30 + "#op30-70/71; typed argument ownership",
            "Retain CLI changes as behavior/interface audit. The six removed invalid successful scenarios are covered by explicit negative tests; removed padded-vocab-size on FSDP is a real rejection change even though it was unused by FSDP execution.",
        )
    if category == "scope/controller-excludes-training-backend":
        return _EvidenceRule(
            "scope/controller-no-training-backend",
            "role-boundary-with-owner-witness",
            "InferenceControllerConfig and MultiLoraConfig intentionally omit RawTrainerBackendConfig. Their specs now slice those exact component models instead of sharing the complete parser Namespace. The audit searches the same recorded case for the original typed value in an actual backend/raw-backend owner, and reports missing witnesses separately; it never equates omission with equality.",
            [
                "miles/utils/args/runtime.py:InferenceControllerConfig",
                "miles/utils/args/runtime.py:MultiLoraConfig",
                "miles/ray/specs/inference.py",
                "miles/utils/test_utils/adhoc_snapshot_comparator/snapshot_scope_evidence.py",
            ],
            _M30 + "#component ownership; scaling config must leave long-lived components",
            "Use the per-difference retained-owner witness and the complete removed-field-consumers-002.json source inventory; any absent witness remains open.",
        )
    if category == "lifecycle/endpoints-not-copied-to-trainer":
        return _EvidenceRule(
            "scope/endpoint-state-owner",
            "endpoint-owner-lifecycle",
            "Trainer configs are constructed before rollout endpoint discovery and no longer receive mutated copies of shared args. RolloutExecutor and InferenceController discover/use router and session endpoints through providers. Training actors do not issue rollout HTTP requests through these copies; their weight-update targets arrive through the explicit inference-controller protocol.",
            [
                "miles/ray/rollout/rollout_executor.py:init",
                "miles/ray/rollout/router_manager.py:resolve_router_addrs",
                "miles/ray/train/group.py:update_weights",
                "miles/utils/args/runtime.py:TrainerConfig",
            ],
            _M30 + "#component ownership and hot restart configuration boundaries",
            "Match the exact retained rollout endpoint values at router_endpoints/session_endpoints; do not approve arbitrary new hosts or ports.",
        )
    if category == "coverage/missing-case":
        return _EvidenceRule(
            "capture/unpaired-case",
            "unpaired-input-coverage",
            "This whole case exists on only one side. The audit inventory distinguishes six intentionally rejected parser fixtures, three added AMD launcher outputs, and fifteen historical runtime goldens without a paired before capture. Missing execution evidence is not a zero-difference result.",
            [
                "tests/fast/argument_snapshot/argparser/scenarios.py",
                "tests/manual/launch_scripts/test_py_launch_scripts.py",
                "scripts/amd/run_gpt_oss_20b_lora.py",
            ],
            _M30 + "#SoT: 如何处理snapshot test; approved op30-129 invalid scenarios",
            "For each named runtime case, record whether source was deleted, disabled or nightly-only; obtain a paired historical capture before asserting equivalence. Negative parser tests already exercise the rejected paths.",
        )
    if category == "value/list-change" or field.endswith(".txt"):
        return _EvidenceRule(
            "capture/structured-interface-change",
            "lossless-structure-and-cpu-spec-audit",
            "Lists and rendered commands are retained whole in the strict comparison. Separate lossless inventories preserve each parser action keyed by (dest, option_strings), all defaults/const/actions/order, and all 39 chart/command leaves. Chart WORKER_ARGV is incomplete relative to SCENARIO_ARGV. The CPU observer therefore captures actual old ctor closure args and new spec.args for all four worker types. Production decoding of eight historical payloads exposes only empty tuple/list representation changes and three router declarations present on both revisions in the replay environment but absent historically; these remain strict differences.",
            [
                "tests/fast/argument_snapshot/argparser/scenarios.py",
                "tests/fast/charts/miles_run/test_snapshot.py:SCENARIO_ARGV",
                "tests/fast/charts/miles_run/test_snapshot.py:WORKER_ARGV",
                "tests/fast/utils/external_utils/command_utils/helm_backend/launcher/test_launch_snapshot.py",
            ],
            _M30 + "#SoT: 如何处理snapshot test; op30-98/169",
            "Retain parser-action-analysis-001.json, command-and-chart-analysis-001.json, chart-before/after-001.json and chart payload evidence. The incomplete/fake old fixtures and historical replay environment difference limit what the golden alone proves; the separate real-spec observation closes the configuration-construction comparison.",
        )
    if field == "add_argument":
        return _EvidenceRule(
            "schema/parser-instance-method",
            "parser-implementation-state",
            "The old SGLang registration temporarily overwrote parser.add_argument and restored the bound class method by assigning it to the instance. Its instance __dict__ therefore retained that method. New registration does not leave this shadow attribute; method resolution still reaches argparse._ActionsContainer.add_argument. Registered actions and defaults are separately compared losslessly.",
            [
                "before:miles/backends/sglang_utils/arguments.py:_add_prefixed_server_args",
                "after:miles/backends/sglang_utils/sglang_config.py:_add_prefixed_primary_server_args",
                "tests/fast/argument_snapshot/argparser/schema.py:snapshot_parser",
            ],
            _M30 + "#explicit mechanical representation changes",
            "Require the exact old bound-method identity and only the parser.state.add_argument path; retain the representation difference in strict output.",
        )
    if field in {"cp_comm_type", "padded_vocab_size"}:
        return _EvidenceRule(
            "schema/inactive-megatron-shim",
            "inactive-backend-declaration-removal",
            "FSDP and TorchTitan snapshots carried null Megatron-only parser shim fields. Their execution paths do not read these Namespace attributes; cp_comm_type uses elsewhere refer to Megatron ParallelState, and padded vocabulary is owned by Megatron tokenizer/model setup. Active Megatron values are separately moved and compared in backend configuration.",
            [
                "miles/utils/args/configs/custom_megatron_plugins.py",
                "miles/backends/fsdp_utils/arguments.py",
                "miles/utils/test_utils/adhoc_snapshot_comparator/snapshot_ownership.py",
            ],
            _M30 + "#backend ownership",
            "Require train_backend in {fsdp, torchtitan} and old value None for each removed shim; any active/non-null case needs its own consumer evidence.",
        )
    if field == "eval_sglang_enable_metrics":
        return _EvidenceRule(
            "derived/eval-server-overrides",
            "old-resolver-replay",
            "The old eval-only flat flag is consumed into the eval model server-group overrides. The pinned SGLang replay strictly compares every model/group override; both fsdp_eval_fleet and megatron_eval_fleet retain enable_metrics=True.",
            [
                "miles/backends/sglang_utils/arguments.py:collect_eval_sglang_overrides",
                "miles/backends/sglang_utils/sglang_config.py:resolve_sglang_config",
            ],
            _M30 + "#op30-135",
            "Cross-reference the exact parsed_config row in sglang-replay-001.json; do not drop eval overrides from comparison.",
        )
    if field == "sglang_scaling":
        return _EvidenceRule(
            "ownership/parsed-sglang-scaling",
            "old-resolver-replay",
            "All 107 paired parsed configs and 124 runtime WorkerManager records exactly match the old resolver's num_gpus, gpu_offset and engine_offset extracted by model/group index. This verifies both the parser split and the captured runtime scaling owner, while each removed component copy requires its same-generation owner witness.",
            [
                "miles/backends/sglang_utils/sglang_config.py:ServerGroupConfig.resolve",
                "miles/utils/test_utils/adhoc_snapshot_comparator/snapshot_sglang_evidence.py:_split_models",
            ],
            _M30 + "#op30-135",
            "Distinguish same-generation runtime owner evidence from parser-only evidence; do not infer actual allocation from an uncaptured deployment boundary.",
        )
    if field == "rollout_indexer_topk_num_streams":
        return _EvidenceRule(
            "schema/optional-indexer-streams",
            "inactive-default-declaration",
            "The old Megatron-only derived value now belongs to a shared backend trait. Existing Megatron values are explicitly copied and strictly compared; FSDP's newly serialized value is None. The response decoder previously used getattr(args, rollout_indexer_topk_num_streams, None); direct access to the explicit None preserves the absent-value fallback and the same optional stream-count assertion.",
            [
                "miles/utils/args/configs/backend_fields.py:TrainerBackendTraitConfig",
                "miles/backends/sglang_utils/sglang_config.py",
            ],
            _M30 + "#backend trait ownership",
            "Require the added value is None and verify the conditional consumer; any non-null addition remains unverified.",
        )
    if field == "requested_load":
        return _EvidenceRule(
            "lifecycle/orchestrator-requested-load",
            "owner-state-not-worker-input",
            "requested_load records the original checkpoint target for orchestration/hot restart, not the resolved per-worker load request. Worker config now omits it; the orchestrator retains it and creates explicit per-trainer requests. The pre-existing critic hot-restart TODO is not introduced by this refactor.",
            [
                "miles/backends/megatron_utils/megatron_config.py:resolve_args_checkpoint_load",
                "miles/utils/args/trainer_utils.py:compute_trainer_ckpt_load_config",
                "miles/utils/hot_restart.py",
            ],
            _M48 + "#op48-130/131/132",
            "Require a retained orchestrator witness or documented request lifecycle; checkpoint_load outputs remain strictly compared.",
        )
    if category == "scope/parser-only-input":
        return _EvidenceRule(
            "lifecycle/consumed-parser-input",
            "input-consumed-before-slicing",
            "custom_config_path is read during argument parsing before typed configuration is built; its resolved fields remain compared. megatron_config is parsed into raw_megatron.trainers rather than carried as YAML/base64 text. Neither path is a runtime consumer input after resolution.",
            [
                "miles/utils/arguments.py:parse_args_and_get_parser",
                "miles/backends/megatron_utils/megatron_config.py:resolve_megatron_config",
                "miles/utils/test_utils/adhoc_snapshot_comparator/snapshot_megatron_evidence.py",
            ],
            _M30 + "#typed configuration ownership",
            "For the two custom YAML cases compare all resulting fields; for all 18 non-null Megatron inputs use the pinned old resolver replay, including its six critic lifecycle differences.",
        )
    if field == "trainer_init_expected_num_cells":
        return _EvidenceRule(
            "lifecycle/trainer-startup-target",
            "derived-startup-target",
            "The parser initially stores None; compute_trainer_args resolves the exact old compute_trainer_num_cells formula before slicing. TrainerController then reads this stored startup target. An explicit new override intentionally enables elastic scaling without changing long-lived configs.",
            [
                "miles/backends/megatron_utils/megatron_config.py:_resolve_trainer_init_expected_num_cells",
                "miles/ray/train/group.py:_expected_num_cells",
                "miles/utils/args/trainer_utils.py:compute_trainer_total_gpus",
            ],
            _M30 + "#op30-133; op30-165",
            "Replay the old role-specific GPU count formula against every captured trainer scalar; None outside trainer scope is an unused declaration, not a deployed target.",
        )
    if field in {"trainer_role", "trainer_actor_index"}:
        return _EvidenceRule(
            "ownership/trainer-identity",
            "explicit-constructor-state",
            "Role and actor order were constructor inputs and loop position before. They are now explicit serialized fields, assigned from the same resolved trainer record before constructing TrainerController. Critic index remains None; actor indices follow original actor order.",
            [
                "miles/backends/megatron_utils/megatron_config.py:compute_trainer_args",
                "miles/ray/train/group.py:TrainerController.__init__",
                "miles/ray/specs/train.py:TrainerControllerSpec",
            ],
            _M30 + "#trainer configuration ownership",
            "Compare per-process identity with the old resolved trainer list in snapshot_megatron_evidence, including all critic and multi-policy cases.",
        )
    if field == "inference_runtime_mut_state":
        return _EvidenceRule(
            "runtime/observed-inference-fleet",
            "intentional-dynamic-fleet-read",
            "The three zero counters are pre-initialization state, not zero allocated GPUs. RolloutExecutor.init refreshes them from InferenceController before init_http_client. Metrics and HTTP concurrency now use observed engine/GPU counts, including later fleet changes, instead of launch-time scaling inputs. This is a real dynamic-fleet behavior change.",
            [
                "miles/utils/args/component_rollout.py:InferenceRuntimeMutState",
                "miles/ray/rollout/rollout_executor.py:init",
                "miles/ray/rollout/inference_controller.py:get_inference_runtime_immut_state",
                "miles/utils/http_utils.py:init_http_client",
                "miles/ray/rollout/metrics.py:_compute_perf_metrics_from_samples",
            ],
            _M30 + "#scaling config must leave orchestrator and rollout executor",
            "Run refresh/init and live-fleet consumer tests. Capture post-refresh counts to prove each historical runtime fleet, rather than approving the zero initializer as equivalent to launch GPU counts.",
        )
    if field == "session_server_instances":
        return _EvidenceRule(
            "runtime/session-endpoints",
            "endpoint-lifecycle",
            "The typed schema explicitly declares None before discovery. wait_session_server_ready publishes the instance list only for the enabled session server, and consumers use it after readiness. Trainer snapshots no longer inherit the rollout endpoint copy.",
            [
                "miles/utils/args/configs/session.py:SessionConfig",
                "miles/ray/rollout/router_manager.py:wait_session_server_ready",
                "miles/ray/rollout/rollout_executor.py:init",
            ],
            _M30 + "#component configuration ownership",
            "For added None verify disabled/pre-readiness case; for removed lists compare the retained rollout/router owner and its discovery stage. Do not discard changed endpoint values.",
        )
    if category == "representation/backend-discriminator":
        return _EvidenceRule(
            "schema/backend-discriminator",
            "explicit-type-tag",
            "The new backend_name is the discriminator of the selected TrainerBackendConfig union. The old flat train_backend selected the same implementation; the discriminator carries no independent training knob.",
            [
                "miles/backends/fsdp_utils/config.py; miles/backends/megatron_utils/megatron_config.py:MegatronArgsNamespace",
                "miles/utils/args/trainer_utils.py",
            ],
            _M30 + "#typed backend configuration",
            "Require exact equality with the captured train_backend, not an arbitrary newly added string.",
        )
    if field == "trainers":
        return _EvidenceRule(
            "derived/megatron-trainers",
            "old-resolver-replay-with-request-lifecycle",
            "The raw megatron_config text is consumed once into trainers. snapshot_megatron_evidence executes the pinned before resolver and adds only actor_index derived from original actor order. Every original override is strictly compared, including critic load paths.",
            [
                "miles/backends/megatron_utils/megatron_config.py:resolve_megatron_config",
                "miles/backends/megatron_utils/megatron_config.py:_compute_trainers",
            ],
            _M30 + "#typed backend configuration",
            "Use the full replay artifact; missing records and any override differences remain failures.",
        )
    if field in {"models", "sglang_config"}:
        return _EvidenceRule(
            "derived/sglang-models",
            "old-resolver-replay",
            "The old raw sglang_config is replaced by its resolved models. A pinned before122 resolver replays all 1600 captured argument records; 1598 paired model lists match exactly after the explicit scaling ownership split, including group order and overrides. The other two cases were removed; the 124 captured runtime scaling owners also match exactly.",
            [
                "miles/backends/sglang_utils/sglang_config.py:resolve_sglang_config",
                "miles/utils/test_utils/adhoc_snapshot_comparator/snapshot_sglang_evidence.py",
            ],
            _M30 + "#op30-135",
            "Link the exact case/stage to sglang-replay-001.json; check the same-generation scaling owner independently of the model list.",
        )
    if category == "scope/scaling-removed-from-component":
        return _EvidenceRule(
            "capture/scaling-owner",
            "owner-witness-required",
            "ScalingConfig is omitted from long-lived component configs, but RayWorkerManager logs the full AllConfig before slicing ScalingConfig. The audit must compare each removed typed value with that actual owner in the same run/deployment generation; a missing component copy is not a missing owner.",
            [
                "miles/utils/logging_utils.py:configure_logger",
                "miles/utils/args/configs/scaling.py:ScalingConfig",
                "miles/utils/workers/ray_worker_manager.py:RayWorkerManager.init",
            ],
            _M30 + "#SoT: 如何处理snapshot test; op30-131/135",
            "Require an exact typed owner witness in the same generation for every removed value. For any truly missing owner, capture scaling and resolved scheduling at the owner boundary; do not assume all 9569 removed copies require recapture.",
        )
    if category == "behavior/kimi-checkpoint-recipe-path":
        return _EvidenceRule(
            "behavior/kimi-layout-cache",
            "historical-behavior-change-stale-after",
            "These captures retain the historical KDA cache suffix introduced by op48-101. Op48-225 (437ecda9ff254e288078654488d3f11500cc5a54) restores the standard BF16 directory for both producer and default ref_load; 137 c97c1f1b contains two freshly recorded launch text goldens and two new H200/wandb runtime variants, while the two canonical runtime YAML goldens retain their historical provenance. This explains historical before/after values only and does not certify the current launcher or tensor equivalence.",
            [
                "historical after137 43d2546a6f4cb5e9095f0f37a4e3f1eba1189ace:scripts/run_kimi_k3.py:_torch_dist_name",
                "scripts/run_kimi_k3.py:ScriptArgs.__post_init__",
                "scripts/run_kimi_k3.py:prepare_torch_dist",
            ],
            _M48 + "#op48-101; op48-225",
            "Keep the strict difference. Use the real prepare_torch_dist and train captures from 137 c97c1f1b and independently verify the H200/wandb variants against raw CI records. The two canonical runtime YAML records remain historical; equivalence still requires same-input before captures. Never rewrite historical path values in the comparator.",
        )
    if category.startswith("lifecycle/checkpoint") or field in {
        "load",
        "critic_load",
        "start_rollout_id",
        "finetune",
        "no_load_optim",
        "no_load_rng",
        "ckpt_step",
    }:
        return _EvidenceRule(
            "lifecycle/checkpoint-request",
            "intentional-lifecycle-change",
            "Before parsing resolved missing load directories by mutating load/finetune/no_load flags and ckpt_step. The new orchestrator preserves requested inputs, deep-copies them into MegatronCkptLoadConfig, and exposes those fields only inside apply(). Ordinary process_config values must differ; compare the actual checkpoint_load stage and returned rollout id instead.",
            [
                "miles/backends/megatron_utils/megatron_config.py:resolve_args_checkpoint_load",
                "miles/utils/args/trainer_utils.py:compute_trainer_ckpt_load_config",
                "miles/backends/megatron_utils/checkpoint_request.py:MegatronCkptLoadConfig.apply",
                "miles/backends/megatron_utils/model.py:load_model_state",
            ],
            _M48 + "#op48-130/131/132",
            "CPU checkpoint_request and trainer_utils tests exercise creation, fallback, sequential requests, exception cleanup and non-mutation. Any checkpoint_load value mismatch remains a failure requiring request-specific inspection.",
        )
    if category == "lifecycle/critic-loss-selected-earlier":
        return _EvidenceRule(
            "lifecycle/critic-loss",
            "observed-training-value-preserved",
            "The old critic assigned value_loss immediately before train_first_step. compute_trainer_args now assigns it before process_config and checkpoint_load. Only three earlier-stage values differ; no captured train_first_step loss_type differs.",
            [
                "before:miles/backends/megatron_utils/actor.py:_train_critic",
                "after:miles/backends/megatron_utils/megatron_config.py:compute_trainer_args",
                "miles/backends/training_utils/loss/objective.py",
            ],
            _M30 + "#trainer configuration ownership",
            "Retain the earlier-stage difference; verify all critic train_first_step records, rather than declaring policy_loss equal to value_loss.",
        )
    if category == "lifecycle/initial-cell-count-derived-earlier":
        return _EvidenceRule(
            "lifecycle/initial-cells",
            "old-cell-count-replay-verified",
            "For owned inference engines the old rollout server derived model.num_server_cells at startup; the new parser stores the same per-model count earlier. Explicit user thresholds and external providers remain separate branches. The pinned old resolver replay checks every captured derived count.",
            [
                "before:miles/ray/rollout/rollout_server.py:_compute_init_expected_num_cells",
                "after:miles/utils/arguments.py:_compute_init_expected_num_cells",
                "after:miles/ray/rollout/rollout_server.py:_compute_init_expected_num_cells",
            ],
            _M30 + "#op30-165",
            "Join this concrete path to sglang-replay evidence; reject mismatches and do not equate None with a count map.",
        )
    if category == "representation/numeric-type-change":
        return _EvidenceRule(
            "representation/typed-numeric-input",
            "consumer-arithmetic-preserved",
            "Only min_batch_collection_ratio (1 to 1.0), tis_clip_low (0 to 0.0), and critic kl_coef (0 to 0.0) occur. The first has no in-repo consumer; the other two feed unchanged clamp/comparison/KL tensor arithmetic. CPU production-function tests verify their outputs. Serialized types remain strictly different.",
            [
                "miles/utils/args/configs/rollout_buffer.py:RolloutBufferConfig",
                "miles/utils/args/configs/algo.py:AlgoConfig",
                "miles/backends/training_utils/loss/hub/corrections.py",
                "miles/backends/training_utils/loss/hub/math_utils.py",
            ],
            _M30 + "#typed configuration; exact-type snapshot comparison",
            "Do not normalize int and float; keep the typed difference with the consumer evidence.",
        )
    if category == "representation/wandb-generated-value":
        return _EvidenceRule(
            "capture/credential-normalization",
            "capture-boundary-incompatible",
            "The old capture stored a credential hash; the new generated-value registry replaces the exact registered environment credential with a symbolic token. wandb_utils still passes the same args.wandb_key to wandb.login, but these two persisted representations cannot prove the two runs used the same credential.",
            [
                "miles/utils/audit_utils/config_snapshot/normalizer.py:_normalize_field",
                "miles/utils/audit_utils/config_snapshot/generated_value_registry.py",
                "miles/utils/tracking_utils/wandb_utils.py:init_wandb",
            ],
            _M30 + "#SoT: 如何处理snapshot test",
            "Use the same non-secret credential fixture and capture policy on both boundaries; preserve this failure until paired evidence exists. Do not reconstruct secrets or convert a hash into an assumed token.",
        )
    if field == "lora_B_init_method":
        return _EvidenceRule(
            "representation/explicit-lora-zero",
            "consumer-default-preserved",
            "Both old LoRA consumers used getattr(args, lora_B_init_method, zero); the new parser materializes zero and the same consumers use direct access. The captured added value is the exact previous consumer fallback, but adding it to before would violate the snapshot rule.",
            [
                "before:miles/backends/megatron_utils/lora/slots.py:create_multi_lora_instance",
                "before:miles/backends/megatron_utils/lora/utils.py",
                "after:miles/utils/arguments.py:parse_args_and_get_parser",
            ],
            _M30 + "#explicit field ownership",
            "Keep the missing-versus-present difference; reject any value other than the documented zero fallback.",
        )
    if field == "offload":
        return _EvidenceRule(
            "representation/consumed-offload-alias",
            "consumer-inputs-preserved",
            "Both argument validators consume --offload into offload_train/offload_rollout and delete the alias. The typed schema materializes the unused alias as false; runtime consumers read the two resolved fields, which remain in the strict comparison.",
            [
                "before:miles/utils/arguments.py:validate_args",
                "after:miles/utils/arguments.py:validate_args",
                "miles/utils/args/configs/cluster.py:ClusterConfig",
            ],
            _M30 + "#typed configuration",
            "Check the actual resolved offload_train/offload_rollout values and never use the new false alias to overwrite before.",
        )
    if field in {"hf_lora_targets", "lora_adapter_targets"}:
        return _EvidenceRule(
            "representation/optional-lora-targets",
            "disabled-branch-representation",
            "The old validator only assigned resolved target lists when LoRA was enabled. Typed LoraConfig declares absent targets as None; enabled target lists are still strictly compared (only the two separately human-approved parsed lists are sorted).",
            [
                "miles/utils/lora/arguments.py:validate_lora_args",
                "miles/utils/lora/utils.py:is_lora_enabled",
                "miles/utils/args/configs/lora.py:LoraConfig",
            ],
            _M30 + "#SoT: 如何处理snapshot test; explicit LoRA list approval is separate",
            "Verify the added value is None and the owning case has LoRA disabled; any non-None added target requires separate investigation.",
        )
    if field in {"num_layers", "custom_agent_function_path"}:
        return _EvidenceRule(
            "representation/explicit-optional-consumer-input",
            "consumer-default-preserved",
            "The new serialized None is the old consumer fallback: SessionServerConfig used getattr(args, num_layers, None), and call_agent_abort_hook used getattr(args, custom_agent_function_path, None) followed by an immediate return. Active custom functions and model-layer values remain strict comparisons; no absent field is populated by the comparator.",
            [
                "before:miles/rollout/session/config.py:compute_session_server_config",
                "before:miles/utils/misc.py:call_agent_abort_hook",
                "after:miles/utils/args/configs/rollout.py",
            ],
            _M30 + "#explicit configuration fields",
            "Require exactly added None. A non-null addition is not covered by this fallback argument.",
        )
    if field == "raw_megatron":
        return _EvidenceRule(
            "derived/non-megatron-trainer-description",
            "old-resolver-replay-with-request-lifecycle",
            "FSDP and TorchTitan also used resolve_megatron_config to describe actor/critic roles before; the historically named container now stores that result explicitly alongside an empty Megatron base_args dictionary. The selected actual training backend remains train_backend and raw_fsdp/raw_torchtitan.",
            [
                "miles/ray/specs/train.py",
                "miles/backends/megatron_utils/megatron_config.py:resolve_megatron_config",
                "miles/utils/args/trainer_utils.py:compute_trainer_config",
            ],
            _M30 + "#typed trainer configuration",
            "The all-backend old resolver replay must match these lists; the backend discriminator witness separately proves the training implementation did not switch.",
        )
    if field in {"raw_fsdp", "raw_torchtitan"}:
        return _EvidenceRule(
            "representation/declared-optional-field",
            "case-precondition-required",
            "A typed optional field is now serialized where the old Namespace lacked it. No equality is inferred from the presence change; the field-specific inactive backend or missing optional hook branch must be checked against the same case.",
            [
                "miles/utils/args/configs/backend_fields.py:RawTrainerBackendConfig",
                "miles/utils/args/trainer_utils.py:compute_trainer_config",
                "miles/utils/args/configs/rollout.py",
            ],
            _M30 + "#configuration ownership",
            "Check the concrete added value is None and the selected backend/hook does not consume it. Do not populate before defaults.",
        )
    if field.startswith("tinker_"):
        return _EvidenceRule(
            "representation/tinker-config",
            "feature-boundary-change",
            "Tinker server and LoRA group configuration is now explicit on AllConfig while absent from these old snapshots. The multi_lora branch and actual values must be checked; absence is not automatically equivalent to the newly declared default.",
            [
                "miles/utils/args/configs/tinker.py:TinkerConfig",
                "miles/utils/lora/utils.py:is_multi_lora_enabled",
                "miles/utils/args/configs/tinker.py; serve_tinker.py",
            ],
            _M30 + "#multi-LoRA current implementation boundary",
            "Verify multi_lora is disabled for each added default; enabled cases require the concrete Tinker execution configuration.",
        )
    return _EvidenceRule(
        category + "/" + field,
        "requires-specific-witness",
        "This exact typed pair remains a strict failure. Its detailed owner, consumer or replay witness must be attached before making a non-bug claim; the category alone proves nothing.",
        [],
        _M30 + "#SoT: 如何处理snapshot test",
        "Inspect this group's complete variants and corresponding fixed-source consumer; never approve by field name alone.",
    )
