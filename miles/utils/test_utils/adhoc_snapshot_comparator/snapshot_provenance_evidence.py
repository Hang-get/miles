import hashlib
import json
import logging
from pathlib import Path

from pydantic import JsonValue, ValidationError

from miles.utils.file_utils import atomic_write_text
from miles.utils.pydantic_utils import FrozenStrictBaseModel
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity


logger = logging.getLogger(__name__)


def _load_provenance(*, path: Path | None, after_directory: Path | None = None) -> dict[str, dict[str, JsonValue]]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    assert after_directory is not None, "Provenance requires the actual after inventory"
    assert (
        data["after_inventory"]["sha256"] == _snapshot_identity(directory=after_directory)["sha256"]
    ), "Stale provenance inventory"
    assert data["normalizer_sha256"] == _normalizer_hash(), "Changed normalizer source"
    result = {}
    for row in data["proof"]:
        difference = SnapshotDifference.model_validate({name: row[name] for name in ("path", "before", "after")})
        witness = row["witness"]
        assert witness["full_normalized_stage_matches"] is True
        assert difference.path[-1] == "wandb_key"
        assert difference.before.present and difference.after.present
        assert difference.after.value == "$WANDB_KEY_environment"
        assert witness["registered_kind"] == "wandb_key" and witness["registered_name"] == "environment"
        assert witness["registered_redacted_value"] == difference.before.value
        assert isinstance(difference.before.value, str) and difference.before.value.startswith("redacted-sha256:")
        assert len(witness["sha256"]) == 64 and all(char in "0123456789abcdef" for char in witness["sha256"])
        point = witness["point"]
        assert difference.path[3] == f"{point['stage']}-{point['index']:04d}"
        key = json.dumps(difference.path)
        assert key not in result
        result[key] = row
    return result


def _provenance_evidence(
    *, difference: SnapshotDifference, provenance: dict[str, dict[str, JsonValue]]
) -> dict[str, JsonValue] | None:
    row = provenance.get(json.dumps(difference.path))
    if row is None:
        return None
    pair = difference.model_dump(include={"before", "after"})
    if json.dumps(pair, sort_keys=True) != json.dumps(
        {name: row[name] for name in ("before", "after")}, sort_keys=True
    ):
        return None
    # Each witness binds the exact case/process/stage and old redacted digest
    # to an actual after record registering that same digest as environment.
    # This proves the normalizer's representation change, never raw secrets.
    # Evidence cannot approve or remove the strict value difference.
    return {
        "rule": "provenance/wandb-environment-digest",
        "verdict": "raw-redacted-value-and-origin-verified",
        "source": "after137:miles/utils/audit_utils/config_snapshot/normalizer.py:_normalize_field",
        "reason": "The exact old digest occurs in the matching after raw capture and its generated-value registry. The normalizer deterministically turns precisely that registered digest into the environment token; unrelated keys are not normalized.",
        "witness": row["witness"],
    }


def _build_provenance(
    *,
    before: Path,
    after: Path,
    raw_directory: Path,
    output: Path,
    original: dict[str, JsonValue] | None = None,
    actual: dict[str, JsonValue] | None = None,
) -> None:
    from collections import defaultdict

    from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotRecord
    from miles.utils.audit_utils.config_snapshot.normalizer import (
        collect_generated_values,
        normalize_record,
        normalized_source_name,
    )
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences, _lookup
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory

    assert not output.exists()
    original = expand_snapshot_directory(directory=before) if original is None else original
    actual = expand_snapshot_directory(directory=after) if actual is None else actual
    candidates = defaultdict(list)
    rejected_attempts = []
    for attempt_path in sorted(raw_directory.rglob("attempt.json")):
        attempt_data = json.loads(attempt_path.read_text())
        case, completed = _attempt_case(data=attempt_data)
        if not completed:
            continue
        paths = sorted((attempt_path.parent / "records").glob("*.json"))
        try:
            records = [ConfigSnapshotRecord.model_validate_json(path.read_text()) for path in paths]
        except ValidationError as error:
            logger.warning(
                "Historical provenance schema requires a fixed-source replay: %s", attempt_path, exc_info=True
            )
            rejected_attempts.append(
                {
                    "attempt": str(attempt_path),
                    "verdict": "historical-schema-replay-required",
                    "error": str(error),
                    "raw_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                }
            )
            continue
        generated = collect_generated_values(records)
        for path, record in zip(paths, records, strict=True):
            ctx = record.context
            process = (
                f"{ctx.name}/{ctx.deploy_component}/{ctx.deploy_instance_id}/{normalized_source_name(ctx.source)}"
            )
            scope = [case, process, "stages", record.point.to_key()]
            observed = _lookup(value=actual, path=scope)
            normalized = normalize_record(record, generated_values=generated[ctx.generation])
            if not observed.present or _differences(before=normalized, after=observed.value, path=[]):
                continue
            key_path = [*scope, "args", "wandb_key"]
            old = _lookup(value=original, path=key_path)
            new = _lookup(value=actual, path=key_path)
            if not old.present or new.value != "$WANDB_KEY_environment":
                continue
            registry = [
                entry
                for entry in generated[ctx.generation]
                if entry.kind == "wandb_key" and entry.name == "environment"
            ]
            assert len(registry) == 1, (scope, registry)
            candidates[json.dumps(key_path)].append(
                {
                    "path": key_path,
                    "before": old.model_dump(),
                    "after": new.model_dump(),
                    "witness": {
                        "record": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "point": record.point.model_dump(),
                        "registered_kind": registry[0].kind,
                        "registered_name": registry[0].name,
                        "registered_redacted_value": registry[0].value,
                        "full_normalized_stage_matches": True,
                    },
                }
            )
    proof, conflicts = [], []
    for rows in candidates.values():
        if all(row["witness"]["registered_redacted_value"] == row["before"]["value"] for row in rows):
            proof.append(rows[0])
        else:
            conflicts.append(rows)
    atomic_write_text(
        path=output,
        text=json.dumps(
            {
                "normalizer_sha256": _normalizer_hash(),
                "after_inventory": _snapshot_identity(directory=after),
                "before_inventory": _snapshot_identity(directory=before),
                "proof": proof,
                "ambiguous_or_changed_digests": conflicts,
                "rejected_attempts": rejected_attempts,
            },
            indent=2,
        ),
    )


class _HistoricalAttempt(FrozenStrictBaseModel):
    test: str
    completed: bool


def _attempt_case(*, data: dict[str, JsonValue]) -> tuple[str, bool]:
    from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotTestAttempt

    # Absence of variant is an explicit historical schema, not a default
    # hardware/tracking identity. New attempts retain their full variant path.
    if "variant" not in data:
        historical = _HistoricalAttempt.model_validate(data, strict=True)
        return "runtime_config/" + historical.test.removesuffix(".py") + ".yaml", historical.completed
    attempt = ConfigSnapshotTestAttempt.model_validate(data, strict=True)
    return (
        "runtime_config/variants/" + attempt.variant.to_key() + "/" + attempt.test.removesuffix(".py") + ".yaml",
        attempt.completed,
    )


def _normalizer_hash() -> str:
    from miles.utils.audit_utils.config_snapshot import normalizer

    return hashlib.sha256(Path(normalizer.__file__).read_bytes()).hexdigest()


def _main() -> None:
    from typing import Annotated

    import typer

    def build(
        before: Annotated[Path, typer.Option()],
        after: Annotated[Path, typer.Option()],
        raw_directory: Annotated[Path, typer.Option()],
        output: Annotated[Path, typer.Option()],
    ) -> None:
        _build_provenance(before=before, after=after, raw_directory=raw_directory, output=output)

    typer.run(build)


if __name__ == "__main__":
    _main()
