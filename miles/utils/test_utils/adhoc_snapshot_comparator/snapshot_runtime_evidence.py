import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences, _lookup
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_sglang_evidence import _inputs

app = typer.Typer(add_completion=False)


@app.command()
def audit(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    variants: Annotated[Path | None, typer.Option()] = None,
) -> None:
    assert not output.exists()
    original = expand_snapshot_directory(directory=before)
    actual = expand_snapshot_directory(directory=after)
    variant_identity = None
    if variants is not None:
        from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_variant_audit import _diagnostic_views

        saved = json.loads(variants.read_text())
        assert saved["before_inventory"]["sha256"] == _snapshot_identity(directory=before)["sha256"]
        assert saved["after_inventory"]["sha256"] == _snapshot_identity(directory=after)["sha256"]
        views = _diagnostic_views(rows=saved["variants"], original=original, actual=actual)
        original, actual = views.original, views.actual
        variant_identity = {"sha256": hashlib.sha256(variants.read_bytes()).hexdigest(), "aliases": views.aliases}
        del saved, views
    records = []
    for path, args in _inputs(snapshots=original):
        target = _lookup(value=actual, path=[*path, "trainer_init_expected_num_cells"])
        if not target.present or target.value is None:
            continue
        assert type(target.value) is int, (path, target)
        trainer_id = args["trainer_id"]
        assert isinstance(trainer_id, str)
        role = trainer_id.rsplit("-", 1)[-1]
        assert role in {"actor", "critic"}, trainer_id
        expected = _before_trainer_cells(args=args, role=role)
        role_observed = _lookup(value=actual, path=[*path, "trainer_role"])
        differences = _differences(before=expected, after=target.value, path=[])
        records.append(
            {
                "path": path,
                "trainer_id": trainer_id,
                "role": role,
                "role_match": role_observed.present and role_observed.value == role,
                "expected_cells": expected,
                "observed_cells": target.value,
                "match": not differences,
                "inputs": {name: args[name] for name in _input_names(args=args, role=role)},
            }
        )
    counts = {
        "inputs": len(records),
        "cells_match": sum(row["match"] for row in records),
        "roles_match": sum(row["role_match"] for row in records),
    }
    atomic_write_text(
        path=output,
        text=json.dumps(
            {
                "before_commit": "4a8ed9962cd9e878565d748a0dbdd196cab958da",
                "variant_identity": variant_identity,
                "before_inventory": _snapshot_identity(directory=before),
                "after_inventory": _snapshot_identity(directory=after),
                "source": "before:miles/ray/specs/train.py:181-188 and miles/utils/megatron_args_utils.py:compute_megatron_world_size_except_dp",
                "records": records,
                "counts": counts,
            },
            indent=2,
        ),
    )
    if any(not row["match"] or not row["role_match"] for row in records):
        raise typer.Exit(code=1)


def _before_trainer_cells(*, args: dict[str, Any], role: str) -> int:
    # Exact before122 compute_trainer_num_cells formula: independent DP uses
    # the original role's GPU count divided by TP*PP*CP; other modes use one.
    assert role in {"actor", "critic"}
    if not args["indep_dp"]:
        return 1
    total_gpus = args[f"{role}_num_nodes"] * args[f"{role}_num_gpus_per_node"]
    group_size = (
        args["tensor_model_parallel_size"] * args["pipeline_model_parallel_size"] * args["context_parallel_size"]
    )
    return total_gpus // group_size


def _input_names(*, args: dict[str, Any], role: str) -> list[str]:
    return ["indep_dp"] + (
        [
            f"{role}_num_nodes",
            f"{role}_num_gpus_per_node",
            "tensor_model_parallel_size",
            "pipeline_model_parallel_size",
            "context_parallel_size",
        ]
        if args["indep_dp"]
        else []
    )


if __name__ == "__main__":
    app()
