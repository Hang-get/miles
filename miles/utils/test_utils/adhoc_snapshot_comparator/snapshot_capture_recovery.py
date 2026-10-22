import hashlib
import json
import logging
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.serialization import dump_config_snapshot
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import _expand_runtime, expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_provenance_evidence import _attempt_case

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


@app.command()
def audit(
    raw_directory: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    historical_replay: Annotated[Path | None, typer.Option()] = None,
) -> None:
    assert not output.exists(), f"Report already exists: {output}"
    actual = expand_snapshot_directory(directory=after)
    replay = {} if historical_replay is None else json.loads(historical_replay.read_text())
    replayed = {row["attempt"]: row for row in replay.get("attempts", [])}
    results = []
    for path in sorted(raw_directory.rglob("attempt.json")):
        metadata = json.loads(path.read_text())
        name, completed = _attempt_case(data=metadata)
        files = sorted((path.parent / "records").glob("*.json"))
        row = {
            "attempt": str(path),
            "test": metadata["test"],
            "completed": completed,
            "raw_files": [
                {"path": str(item), "sha256": hashlib.sha256(item.read_bytes()).hexdigest()} for item in files
            ],
        }
        if not completed:
            row["verdict"] = "incomplete-attempt-retained"
            results.append(row)
            continue
        if path.as_posix() in replayed and "expanded" in replayed[path.as_posix()]:
            saved = replayed[path.as_posix()]
            _verify_replayed_records(files=files, expected=saved["raw_sha256"])
            assert name in actual
            differences = _differences(before=saved["expanded"], after=actual[name], path=[name])
            row["verdict"] = "exact-historical-after-recovered" if not differences else "historical-after-differs"
            row["differences"] = [
                {**item.model_dump(), "evidence": _recovery_difference_evidence(difference=item)}
                for item in differences
            ]
            row["historical_normalizer_sources"] = replay["source_sha256"]
            results.append(row)
            continue
        try:
            storage = ConfigSnapshotStorage(directory=path.parent / "records")
            case = ConfigSnapshotConverter.convert(storage.read())
        except Exception as error:
            logger.exception("Historical capture cannot be decoded with the current schema: %s", path)
            row["verdict"] = "historical-schema-replay-required"
            row["error"] = f"{type(error).__name__}: {error}"
            results.append(row)
            continue
        if not case.processes or name not in actual:
            row["verdict"] = "no-paired-recorded-case"
            results.append(row)
            continue
        recovered = _expand_runtime(text=dump_config_snapshot(case), directory=after)
        differences = _differences(before=recovered, after=actual[name], path=[name])
        row["verdict"] = "exact-historical-after-recovered" if not differences else "historical-after-differs"
        row["differences"] = [
            {**item.model_dump(), "evidence": _recovery_difference_evidence(difference=item)} for item in differences
        ]
        results.append(row)
    assert results, f"No attempt manifests: {raw_directory}"
    atomic_write_text(
        path=output,
        text=json.dumps({"after_inventory": _snapshot_identity(directory=after), "attempts": results}, indent=2),
    )
    # Recovering an old after capture establishes its provenance only. It
    # cannot fill a missing before record or certify today's source and pins.
    if any(row["verdict"] != "exact-historical-after-recovered" for row in results):
        raise typer.Exit(code=1)


def _recovery_difference_evidence(*, difference: SnapshotDifference) -> dict[str, str]:
    path = difference.path
    if path[-1] in {"load", "critic_load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step"}:
        return {
            "verdict": "earlier-checkpoint-lifecycle-capture",
            "source": "miles/utils/args/trainer_utils.py:compute_trainer_ckpt_load_config; miles/backends/megatron_utils/checkpoint_request.py; M48 op130/131/132",
            "reason": "This attempt predates the final request-scoped checkpoint lifecycle. The exact early resolved value versus final request/presence is retained above; it is not accepted as a matching final capture. A separate later completed attempt reproduces this case's final after exactly.",
        }
    if (
        path[-1] == "start_rollout_id"
        and type(difference.before.value) is int
        and difference.before.value == 0
        and difference.after.value is None
        and difference.after.present
    ):
        return {
            "verdict": "earlier-checkpoint-start-id-mutation",
            "source": "miles/backends/megatron_utils/megatron_config.py:resolve_args_checkpoint_load; miles/backends/megatron_utils/actor.py:init",
            "reason": "The early resolver mutated the shared start ID to zero. The final actor returns the load result's start ID instead of using this captured request field as the final result. This is a capture/lifecycle difference, not a rule equating None with zero.",
        }
    if path[-1] in {"sglang_router_ip", "sglang_router_port", "$tuple"} and "router_endpoints-0000" in path:
        return {
            "verdict": "historical-endpoint-normalizer-change",
            "source": "c78e8715:miles/utils/audit_utils/config_snapshot/endpoints.py; current:miles/utils/audit_utils/config_snapshot/normalizer.py:_normalize_endpoints",
            "reason": "The old endpoint registry emitted owner-qualified host/port tokens; the current generated-value normalizer emits generic HOST/PORT tokens for registered endpoints. These are different normalization contracts, retained literally. This replay alone cannot prove actual endpoint routing equivalence.",
        }
    if (
        path[-1] == "wandb_group"
        and difference.before.value == "$RUN_ID_0001_$CI_COMMIT_NAME_github"
        and difference.after.value == "$RUN_ID_0002_$CI_COMMIT_NAME_github"
    ):
        return {
            "verdict": "different-historical-run-id-allocation",
            "source": "miles/utils/audit_utils/config_snapshot/normalizer.py:_normalize_field; miles/utils/external_utils/command_utils/common.py:create_run_id",
            "reason": "This completed attempt used another registered run-ID allocation order. The final golden matches a different completed attempt exactly. It is not the same run input, and the two group tokens are not equated.",
        }
    return {
        "verdict": "unverified-historical-difference",
        "source": "",
        "reason": "No source-specific rationale is registered; preserve this exact difference as unresolved.",
    }


def _verify_replayed_records(*, files: list[Path], expected: dict[str, str]) -> None:
    observed = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    assert observed == expected, "Historical replay must cover the exact original record inventory"


if __name__ == "__main__":
    app()
