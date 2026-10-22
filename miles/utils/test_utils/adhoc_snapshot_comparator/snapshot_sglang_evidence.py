import hashlib
import importlib.util
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import JsonValue

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.snapshot import snapshot_values
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences, _lookup
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)
_BEFORE_SOURCE_SHA256 = "e9a84ab23769d18d869517e3777619229046379cb6fe3a3b896c0103cec848bf"


@app.command()
def audit(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    before_source: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    variants: Annotated[Path | None, typer.Option()] = None,
) -> None:
    assert not output.exists()
    assert hashlib.sha256(before_source.read_bytes()).hexdigest() == _BEFORE_SOURCE_SHA256
    spec = importlib.util.spec_from_file_location("snapshot_before122_sglang", before_source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Replay the exact before122 resolver, not the new resolver. Its imported
    # WorkerType/file_arg_utils/pydantic_utils are unchanged; the two imported
    # argument helpers retain identical Namespace behavior for these inputs.
    # This is separate behavior evidence, never an equality-transform rule.
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
        try:
            assert "multi_lora" in args
            before_config = module.resolve_sglang_config(Namespace(**args))
            resolved = snapshot_values(before_config)
            models, scaling = _split_models(resolved=resolved)
            observed = _lookup(value=actual, path=[*path, "sglang", "models"])
            differences = [] if not observed.present else _differences(before=models, after=observed.value, path=[])
            observed_scaling = _lookup(value=actual, path=[*path, "sglang_scaling"])
            scaling_differences = (
                []
                if not observed_scaling.present
                else _differences(before=scaling, after=observed_scaling.value, path=[])
            )
            observed_counts = _lookup(value=actual, path=[*path, "init_expected_num_cells"])
            expected_counts = {model.name: model.num_server_cells for model in before_config.models}
            counts_match = isinstance(observed_counts.value, dict) and not _differences(
                before=expected_counts, after=observed_counts.value, path=[]
            )
            records.append(
                {
                    "path": path,
                    "initial_cell_counts": expected_counts,
                    "initial_cell_counts_match": counts_match,
                    "initial_cell_counts_observed": observed_counts.model_dump(),
                    "models_match": observed.present and not differences,
                    "models_differences": [item.model_dump() for item in differences],
                    "models_observed": observed.present,
                    "scaling_match": observed_scaling.present and not scaling_differences,
                    "scaling_observed": observed_scaling.present,
                    "scaling_differences": [item.model_dump() for item in scaling_differences],
                    "before_resolved_models": resolved["models"],
                    "expected_models": models,
                    "expected_scaling": scaling,
                }
            )
        except Exception as error:
            logger.exception("Cannot replay before resolver at %s", path)
            records.append({"path": path, "error": f"{type(error).__name__}: {error}"})
    result = {
        "before_commit": "4a8ed9962cd9e878565d748a0dbdd196cab958da",
        "variant_identity": variant_identity,
        "before_inventory": _snapshot_identity(directory=before),
        "after_inventory": _snapshot_identity(directory=after),
        "before_source_sha256": _BEFORE_SOURCE_SHA256,
        "source_path": "miles/backends/sglang_utils/sglang_config.py",
        "records": records,
        "counts": {
            "inputs": len(records),
            "errors": sum("error" in row for row in records),
            "models_match": sum(row.get("models_match", False) for row in records),
            "initial_cell_counts_match": sum(row.get("initial_cell_counts_match", False) for row in records),
            "scaling_match": sum(row.get("scaling_match", False) for row in records),
            "scaling_not_captured": sum(not row["scaling_observed"] for row in records if "scaling_observed" in row),
        },
    }
    atomic_write_text(path=output, text=json.dumps(result, indent=2))
    if result["counts"]["errors"] or any(not row["models_match"] for row in records if "models_match" in row):
        raise typer.Exit(code=1)


def _inputs(*, snapshots: dict[str, JsonValue]) -> list[tuple[list[str], dict[str, Any]]]:
    result = []
    for case_name, case in snapshots.items():
        if not isinstance(case, dict):
            continue
        if case_name.startswith("parsed_config/") and isinstance(args := case.get("config"), dict):
            result.append(([case_name, "config"], args))
        elif case_name.startswith("runtime_config/"):
            for process_name, process in case.items():
                assert isinstance(process, dict) and isinstance(stages := process["stages"], dict)
                for stage_name, stage in stages.items():
                    assert isinstance(stage, dict) and isinstance(args := stage["args"], dict)
                    result.append(([case_name, process_name, "stages", stage_name, "args"], args))
    return result


def _split_models(*, resolved: dict[str, Any]) -> tuple[list[JsonValue], dict[str, JsonValue]]:
    models = []
    groups = {}
    for model in resolved["models"]:
        retained_groups = []
        scaling_groups = []
        for group in model["server_groups"]:
            # Explicit ownership split from after137 ServerGroupConfig.resolve:
            # every old key survives in exactly one of these two objects.
            scaling_names = {"num_gpus", "gpu_offset", "engine_offset"}
            scaling_groups.append({name: group[name] for name in scaling_names})
            retained_groups.append({name: value for name, value in group.items() if name not in scaling_names})
        groups[model["name"]] = scaling_groups
        models.append(model | {"server_groups": retained_groups})
    return models, {"groups": groups}


if __name__ == "__main__":
    app()
