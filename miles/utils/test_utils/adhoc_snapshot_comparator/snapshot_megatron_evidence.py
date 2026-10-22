import hashlib
import importlib.util
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path
from typing import Annotated, Any

import typer

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences, _lookup
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_input import _load_json
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_sglang_evidence import _inputs
from miles.utils.test_utils.snapshot import snapshot_values

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)
_BEFORE_SOURCE_SHA256 = "227e332d45eaec91cf7a5e2d33cf6fc0ccc38f3a3b53d8d068827cc50c07b07f"


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
    spec = importlib.util.spec_from_file_location("snapshot_before122_megatron", before_source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original = expand_snapshot_directory(directory=before)
    actual = expand_snapshot_directory(directory=after)
    variant_identity = None
    if variants is not None:
        from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_variant_audit import _diagnostic_views

        saved = _load_json(variants.read_text())
        assert saved["before_inventory"]["sha256"] == _snapshot_identity(directory=before)["sha256"]
        assert saved["after_inventory"]["sha256"] == _snapshot_identity(directory=after)["sha256"]
        views = _diagnostic_views(rows=saved["variants"], original=original, actual=actual)
        original, actual = views.original, views.actual
        variant_identity = {"sha256": hashlib.sha256(variants.read_bytes()).hexdigest(), "aliases": views.aliases}
        del saved, views
    records = []
    for path, args in _inputs(snapshots=original):
        try:
            # The old resolver consumes captured original arguments. The added
            # actor index follows the original actor list order, never after.
            resolved = snapshot_values(module.resolve_megatron_config(Namespace(**args)))
            expected = _add_actor_indices(trainers=resolved["trainers"])
            observed = _lookup(value=actual, path=[*path, "raw_megatron", "trainers"])
            differences = [] if not observed.present else _differences(before=expected, after=observed.value, path=[])
            records.append(
                {
                    "path": path,
                    "before_resolved_trainers": resolved["trainers"],
                    "expected_trainers": expected,
                    "observed": observed.present,
                    "match": observed.present and not differences,
                    "differences": [item.model_dump() for item in differences],
                }
            )
        except Exception as error:
            logger.exception("Cannot replay before Megatron resolver at %s", path)
            records.append({"path": path, "error": f"{type(error).__name__}: {error}"})
    counts = {
        "inputs": len(records),
        "errors": sum("error" in row for row in records),
        "match": sum(row.get("match", False) for row in records),
        "not_captured": sum(not row["observed"] for row in records if "observed" in row),
        "different": sum(bool(row.get("differences")) for row in records),
    }
    atomic_write_text(
        path=output,
        text=json.dumps(
            {
                "before_commit": "4a8ed9962cd9e878565d748a0dbdd196cab958da",
                "variant_identity": variant_identity,
                "before_inventory": _snapshot_identity(directory=before),
                "after_inventory": _snapshot_identity(directory=after),
                "before_source_sha256": _BEFORE_SOURCE_SHA256,
                "source_path": "miles/backends/megatron_utils/megatron_config.py",
                "counts": counts,
                "records": records,
            },
            indent=2,
        ),
    )
    if counts["errors"] or counts["different"] or counts["not_captured"]:
        raise typer.Exit(code=1)


def _add_actor_indices(*, trainers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actor_index = 0
    result = []
    for trainer in trainers:
        assert trainer["role"] in {"actor", "critic"}
        assert "actor_index" not in trainer
        index = actor_index if trainer["role"] == "actor" else None
        result.append(trainer | {"actor_index": index})
        actor_index += trainer["role"] == "actor"
    return result


if __name__ == "__main__":
    app()
