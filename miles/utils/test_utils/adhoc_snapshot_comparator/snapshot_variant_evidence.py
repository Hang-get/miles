import hashlib
import json
import re
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.sync import _collect_snapshots
from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _apply_moves, _differences, _sort_lora_targets, _wrap_hooks
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_audit import _category
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotComparisonRules
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_catalog import _describe_difference
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import _expand_runtime, expand_snapshot_directory

app = typer.Typer(add_completion=False)


@app.command()
def audit(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    raw_directory: Annotated[Path, typer.Option()],
    rules: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    assert not output.exists(), f"Report already exists: {output}"
    original = expand_snapshot_directory(directory=before)
    actual = expand_snapshot_directory(directory=after)
    specification = SnapshotComparisonRules.model_validate_json(rules.read_text(), strict=True)
    _apply_moves(expected=original, actual=actual, moves=specification.moves)
    _wrap_hooks(value=original, rules=specification.hook_wraps)
    _sort_lora_targets(value=original, rules=specification.list_sorts)
    rows = _variant_relations(before=original, after=actual)
    regenerated = _collect_snapshots(directory=raw_directory, repo_root=after.parent.parent)
    witnesses = {}
    for path, content in regenerated.items():
        case = str(path.relative_to(after.resolve()))
        expanded = _expand_runtime(text=content, directory=after)
        witnesses[case] = {
            "expanded_differences": [
                item.model_dump() for item in _differences(before=expanded, after=actual[case], path=[case])
            ],
            "golden_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for row in rows:
        row["raw_replay"] = witnesses.get(row["variant_case"])
    atomic_write_text(
        path=output,
        text=json.dumps(
            {
                "mechanical_rules_sha256": hashlib.sha256(rules.read_bytes()).hexdigest(),
                "before_inventory": _snapshot_identity(directory=before),
                "after_inventory": _snapshot_identity(directory=after),
                "attempts": {
                    str(path.relative_to(raw_directory)): json.loads(path.read_text())
                    for path in sorted(raw_directory.rglob("attempt.json"))
                },
                "raw_files": {
                    str(path.relative_to(raw_directory)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(raw_directory.rglob("*.json"))
                },
                "variants": rows,
            },
            indent=2,
        ),
    )
    # A successful replay proves after provenance, not a missing before capture.
    if any(row["raw_replay"] is None or row["raw_replay"]["expanded_differences"] for row in rows):
        raise typer.Exit(code=1)


def _variant_relations(*, before: dict[str, JsonValue], after: dict[str, JsonValue]) -> list[dict]:
    rows = []
    for case, observed in sorted(after.items()):
        if not case.startswith("runtime_config/variants/"):
            continue
        parts = case.split("/")
        assert len(parts) > 5 and parts[3] in {"wandb", "no-wandb"}, f"Invalid variant path: {case}"
        canonical = "/".join([parts[0], *parts[4:]])
        aligned, moves = _align_single_incarnations(original=after.get(canonical), observed=observed)
        differences = _differences(before=aligned, after=observed, path=[canonical]) if canonical in after else []
        before_aligned, before_moves = _align_single_incarnations(original=before.get(canonical), observed=observed)
        before_differences = (
            _differences(before=before_aligned, after=observed, path=[case]) if canonical in before else []
        )
        # The path records a scenario relation only. Hardware, tracking, process
        # ownership and all typed values remain visible; no rename or approval
        # enters the strict before/after comparator.
        rows.append(
            {
                "variant_case": case,
                "hardware": parts[2],
                "tracking": parts[3],
                "canonical_case": canonical,
                "process_identity_correspondences": moves,
                "canonical_after_present": canonical in after,
                "canonical_before_present": canonical in before,
                "exact_variant_before_present": case in before,
                "pairing_verdict": (
                    "exact-variant-before-present" if case in before else "no-explicit-variant-before-key"
                ),
                "canonical_differences": [
                    {
                        **item.model_dump(),
                        "category": _category(difference=item),
                        "historical_rule_candidate": _describe_difference(
                            difference=item, category=_category(difference=item)
                        ),
                    }
                    for item in differences
                ],
                "canonical_difference_count": len(differences),
                "transformed_before_process_correspondences": before_moves,
                "transformed_before_difference_count": len(before_differences) if canonical in before else None,
                "transformed_before_differences": [item.model_dump() for item in before_differences],
            }
        )
    return rows


def _align_single_incarnations(*, original: JsonValue, observed: JsonValue) -> tuple[JsonValue, list[dict]]:
    if not isinstance(original, dict) or not isinstance(observed, dict):
        return original, []
    aligned = dict(original)
    moves = []
    for name in original:
        candidates = [key for key in observed if re.fullmatch(re.escape(name) + r"/[^/]+/incarnation-[0-9]{4}", key)]
        if name in observed or len(candidates) != 1:
            continue
        target = candidates[0]
        assert target not in aligned
        aligned[target] = aligned.pop(name)
        moves.append(
            {
                "canonical_process": name,
                "variant_process": target,
                "source": "op48-246:miles/utils/audit_utils/config_snapshot/converter.py:ConfigSnapshotConverter.convert",
                "reason": "The converter appends the validated cell membership and a real worker incarnation. Exactly one observed incarnation has this original logical owner. Only the key is aligned for diagnostic comparison; every rank, stage, input and typed value is still compared. Multiple incarnations remain unpaired.",
                "verdict": "diagnostic-owner-correspondence-not-before-equivalence",
            }
        )
    return aligned, moves


if __name__ == "__main__":
    app()
