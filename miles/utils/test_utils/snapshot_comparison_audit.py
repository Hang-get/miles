import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.snapshot_comparison_models import SnapshotComparisonReport, SnapshotDifference


def _write_audit(*, report: SnapshotComparisonReport, output: Path) -> None:
    assert not output.exists(), f"Audit already exists: {output}"
    groups: dict[str, dict] = {}
    backend_fields = {
        move.source[-1]
        for move in report.moves
        if any(name in move.target for name in ("backend", "raw_megatron", "raw_fsdp", "raw_torchtitan"))
    }
    categories: Counter[str] = Counter()
    for difference in report.unexplained:
        category = _category(difference=difference)
        if (
            category == "presence/removed-field"
            and difference.path[0].startswith("runtime_config/")
            and len(difference.path) == 6
            and difference.path[1].rsplit("/", 1)[-1] in {"inference_controller", "multi_lora_controller"}
            and difference.path[-1] in backend_fields
        ):
            category = "scope/controller-excludes-training-backend"
        pattern = _pattern(path=difference.path)
        key = json.dumps([category, pattern])
        group = groups.setdefault(key, {"category": category, "pattern": pattern, "count": 0, "variants": {}})
        group["count"] += 1
        categories[category] += 1
        values = difference.model_dump(include={"before", "after"})
        fingerprint = hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()
        variant = group["variants"].setdefault(fingerprint, {"count": 0, "example": difference.model_dump()})
        variant["count"] += 1
    ordered = sorted(groups.values(), key=lambda group: (group["category"], group["pattern"]))
    assert sum(group["count"] for group in ordered) == len(report.unexplained)
    for group in ordered:
        group["variants"] = list(group["variants"].values())
        assert sum(variant["count"] for variant in group["variants"]) == group["count"]
    details = output.with_suffix(".json")
    assert not details.exists(), f"Audit details already exist: {details}"
    atomic_write_text(
        path=details,
        text=json.dumps({"categories": dict(categories), "groups": ordered}, indent=2),
    )
    lines = [
        "## Expanded snapshot comparison audit",
        "",
        f"- Before: `{report.before_directory}`",
        f"- After: `{report.after_directory}`",
        f"- Explicit ownership moves: {len(report.moves):,}",
        f"- Moves with an exactly matching typed destination: {report.verified_move_count:,}",
        f"- Moves with a missing or different destination: {report.unverified_move_count:,}",
        f"- Human-approved differences: {len(report.approved_differences):,}",
        f"- Unresolved strict differences: {len(report.unexplained):,}",
        f"- Unused approvals: {len(report.unused_approvals):,}",
        f"- Exhaustive grouped evidence: `{details.name}` ({len(ordered):,} groups).",
        "- Categories describe evidence, not approval. Every row remains a strict failure until audited.",
        "- Each group retains every distinct typed before/after value pair and an exact source path in the JSON.",
        "- Missing cases, processes, stages and fields remain visible; no defaults or ignored fields are introduced.",
        "",
        "| Evidence category | Strict differences |",
        "| --- | ---: |",
        *[f"| {category} | {count:,} |" for category, count in sorted(categories.items())],
        "",
        "## All residual path groups",
        "",
        "| Category | Path pattern | Differences | Distinct value pairs | Example path |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for group in ordered:
        example = group["variants"][0]["example"]["path"]
        cells = (
            group["category"],
            "/".join(group["pattern"]),
            str(group["count"]),
            str(len(group["variants"])),
            "/".join(example),
        )
        lines.append("| " + " | ".join(cell.replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
    atomic_write_text(path=output, text="\n".join(lines) + "\n")


def _pattern(*, path: list[str]) -> list[str]:
    category = path[0].split("/")[0]
    if category == "runtime_config" and len(path) > 1:
        process = path[1].rsplit("/", 1)[-1]
        process = re.sub(r"\d+", "N", process)
        suffix = [re.sub(r"-\d+$", "-N", part) if index == 3 else part for index, part in enumerate(path[2:], 2)]
        return [category, process, *suffix]
    return [category, *path[1:]]


def _category(*, difference: SnapshotDifference) -> str:
    path = difference.path
    before, after = difference.before, difference.after
    if len(path) == 1 and (not before.present or not after.present):
        return "coverage/missing-case"
    if path[0].startswith("runtime_config/"):
        if len(path) == 2:
            return "coverage/missing-process"
        if len(path) == 4 and path[2] == "stages":
            return "coverage/missing-stage"
        if path[-1] == "ranks":
            return "coverage/rank-membership"
    if not before.present or not after.present:
        if (
            path[0].startswith("runtime_config/")
            and before.present
            and path[-1]
            in {
                "actor_num_nodes",
                "actor_num_gpus_per_node",
                "critic_num_nodes",
                "critic_num_gpus_per_node",
                "rollout_num_gpus",
                "eval_num_gpus",
                "eval_num_gpus_per_engine",
            }
        ):
            return "coverage/scaling-owner-not-captured"
        if path[-1] in {"custom_config_path", "megatron_config"} and before.present:
            return "scope/parser-only-input"
        if path[-1] == "backend_name":
            return "representation/backend-discriminator"
        if path[-1] in {"load", "no_load_optim", "no_load_rng", "finetune", "ckpt_step"}:
            return "lifecycle/checkpoint-field-presence"
        return "presence/added-field" if after.present else "presence/removed-field"
    left, right = before.value, after.value
    field = path[-1]
    if field == "init_expected_num_cells" and left is None and isinstance(right, dict):
        return "lifecycle/initial-cell-count-derived-earlier"
    if field in {"load", "critic_load", "start_rollout_id"} and left is not None and right is None:
        return "lifecycle/checkpoint-resolution-no-longer-mutates-config"
    if (
        field in {"sglang_model_routers", "sglang_router_ip", "sglang_router_port", "session_server_instances"}
        and left is not None
        and right is None
    ):
        return "lifecycle/endpoints-not-copied-to-trainer"
    if (
        field == "wandb_key"
        and isinstance(left, str)
        and left.startswith("redacted-sha256:")
        and right == "$WANDB_KEY_environment"
    ):
        return "representation/wandb-generated-value"
    if field == "loss_type" and left == "policy_loss" and right == "value_loss" and "critic" in path[1]:
        return "lifecycle/critic-loss-selected-earlier"
    if (
        field in {"load", "ref_load"}
        and isinstance(left, str)
        and isinstance(right, str)
        and right == left.removesuffix("_torch_dist") + "-kda-direct-v1_torch_dist"
    ):
        return "behavior/kimi-checkpoint-recipe-path"
    if isinstance(left, str) and isinstance(right, dict) and right.get("path") == left:
        return "representation/hook-path-preserved"
    if type(left) is not type(right):
        if type(left) in (int, float) and type(right) in (int, float) and left == right:
            return "representation/numeric-type-change"
        return "value/type-change"
    if isinstance(left, list):
        return "value/list-change"
    return "value/scalar-change"
