import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_case_evidence import _case_evidence
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_audit import _write_audit
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotComparisonReport,
    SnapshotDifference,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_dependency_evidence import _dependency_evidence
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import (
    _comparison_identity,
    _snapshot_identity,
    _validate_comparison_identity,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_input import _load_json
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import _build_provenance

app = typer.Typer(add_completion=False)


@dataclass
class _DiagnosticViews:
    original: dict[str, JsonValue]
    actual: dict[str, JsonValue]
    differences: list[SnapshotDifference]
    aliases: dict[str, dict[str, str]]


@app.command()
def audit(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    variants: Annotated[Path, typer.Option()],
    rules: Annotated[Path, typer.Option()],
    raw_directory: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    saved = _load_json(variants.read_text())
    assert saved["before_inventory"]["sha256"] == _snapshot_identity(directory=before)["sha256"]
    assert saved["after_inventory"]["sha256"] == _snapshot_identity(directory=after)["sha256"]
    assert saved["mechanical_rules_sha256"] == hashlib.sha256(rules.read_bytes()).hexdigest()
    specification = _load_json(rules.read_text())
    backend_fields = frozenset(
        move["source"][-1]
        for move in specification["moves"]
        if any(name in move["target"] for name in ("backend", "raw_megatron", "raw_fsdp", "raw_torchtitan"))
    )
    del specification
    original = expand_snapshot_directory(directory=before)
    actual = expand_snapshot_directory(directory=after)
    _validate_comparison_identity(identity=saved.get("identity"), actual=actual)
    views = _diagnostic_views(rows=saved["variants"], original=original, actual=actual)
    provenance = output.with_suffix(".provenance.json")
    _build_provenance(
        before=before,
        after=after,
        raw_directory=raw_directory,
        output=provenance,
        original=views.original,
        actual=views.actual,
    )
    del original, actual
    deltas = []
    for row in saved["variants"]:
        for item in row["canonical_differences"]:
            path = [row["variant_case"], *item["path"][1:]]
            path[1] = views.aliases[row["variant_case"]].get(path[1], path[1])
            difference = SnapshotDifference.model_validate(
                {**{key: item[key] for key in ("before", "after")}, "path": path}, strict=True
            )
            evidence = _delta_evidence(difference=difference, actual=views.actual)
            deltas.append({**difference.model_dump(), "evidence": evidence})
    assert len(deltas) == sum(row["canonical_difference_count"] for row in saved["variants"])
    atomic_write_text(
        path=output.with_suffix(".delta.json"),
        text=json.dumps(
            {
                "variants_sha256": hashlib.sha256(variants.read_bytes()).hexdigest(),
                "aliases": views.aliases,
                "assessment_counts": dict(Counter(row["evidence"]["verdict"] for row in deltas)),
                "differences": deltas,
            },
            indent=2,
        ),
    )
    report = SnapshotComparisonReport(
        identity=_comparison_identity(before=before, after=after, rules=rules, actual=views.actual),
        before_directory=str(before.resolve()),
        after_directory=str(after.resolve()),
        moves=[],
        approved_differences=[],
        unexplained=views.differences,
        unused_approvals=[],
        list_sorts=[],
        sorting_differences=[],
    )
    comparison_path = output.with_suffix(".comparison.json")
    assert not comparison_path.exists()
    atomic_write_text(path=comparison_path, text=report.model_dump_json(indent=2))
    _write_audit(
        comparison_path=comparison_path,
        report=report,
        output=output,
        actual=views.actual,
        provenance_path=provenance,
        backend_fields=backend_fields,
        inventory_identity={
            "before": saved["before_inventory"],
            "after": saved["after_inventory"],
            "variants_sha256": hashlib.sha256(variants.read_bytes()).hexdigest(),
            "mechanical_rules_sha256": saved["mechanical_rules_sha256"],
            "scope": "Diagnostic canonical-before to actual variant comparison. Case pairing and unique process aliases are explicit; historical canonical after is a separate operand. Zero moves in this derived report does not mean the saved before operand was untransformed. No strict difference is approved.",
        },
    )
    if report.unexplained or any(row["evidence"]["verdict"] == "unverified" for row in deltas):
        raise typer.Exit(code=1)


def _diagnostic_views(
    *, rows: list[dict], original: dict[str, JsonValue], actual: dict[str, JsonValue]
) -> _DiagnosticViews:
    result = _DiagnosticViews(original={}, actual={}, differences=[], aliases={})
    for row in rows:
        variant, canonical = row["variant_case"], row["canonical_case"]
        aliases = {
            entry["variant_process"]: entry["canonical_process"] for entry in row["process_identity_correspondences"]
        }
        assert len(set(aliases.values())) == len(aliases), "Ambiguous process aliases"
        for entry in row["transformed_before_process_correspondences"]:
            assert (
                aliases.get(entry["variant_process"]) == entry["canonical_process"]
            ), "Before/after logical owners disagree"
        observed = actual[variant]
        assert isinstance(observed, dict)
        assert len({aliases.get(name, name) for name in observed}) == len(observed), "Process alias collision"
        # Only the separately recorded unique owner correspondence changes a
        # lookup key. All captured ranks, stages, values and generations survive.
        result.actual[variant] = {aliases.get(name, name): value for name, value in observed.items()}
        result.aliases[variant] = aliases
        if canonical in original:
            result.original[variant] = original[canonical]
        for item in row["transformed_before_differences"]:
            path = list(item["path"])
            assert path[0] == variant
            path[1] = aliases.get(path[1], path[1])
            result.differences.append(SnapshotDifference.model_validate({**item, "path": path}, strict=True))
    return result


def _delta_evidence(*, difference: SnapshotDifference, actual: dict[str, JsonValue]) -> dict[str, JsonValue]:
    if (evidence := _dependency_evidence(difference=difference)) is not None:
        return evidence
    if (evidence := _case_evidence(difference=difference, actual=actual)) is not None:
        return evidence
    old = "/root/models/Kimi-K3-4layer-64experts-bf16-kda-direct-v1_torch_dist"
    new = "/root/models/Kimi-K3-4layer-64experts-bf16_torch_dist"
    if (
        difference.path[-1] in {"load", "ref_load"}
        and difference.before.present
        and difference.after.present
        and difference.before.value == old
        and difference.after.value == new
        and "/model_scripts/test_kimi_k3_4layer" in difference.path[0]
    ):
        return {
            "rule": "variant/kimi-standard-bf16-path",
            "verdict": "documented-current-launch-input-change",
            "source": "437ecda9ff254e288078654488d3f11500cc5a54:scripts/run_kimi_k3.py:_torch_dist_name; scripts/run_kimi_k3.py:ScriptArgs.__post_init__; c97c1f1b:tests/snapshots/launch_scripts/py/scripts/run_kimi_k3.py",
            "human_sot": "user-written/2026-07-15-impl-M48.md#op48-225",
            "reason": "The real H200/wandb capture and newly recorded producer/train commands use the standard BF16 directory restored by op48-225. The old canonical keeps its historical KDA-suffix provenance. This is an intentional launch input change, not a mechanical normalization or proof of tensor equivalence.",
        }
    return {
        "rule": "variant/unmatched-change",
        "verdict": "unverified",
        "reason": "No exact source-and-value rule covers this observed change; retain the failure and investigate its actual owner and consumer.",
    }


if __name__ == "__main__":
    app()
