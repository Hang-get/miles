import hashlib
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False)
logger = logging.getLogger(__name__)


@app.command()
def replay(
    raw_directory: Annotated[Path, typer.Option()],
    source_directory: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    from miles.utils.audit_utils.config_snapshot.endpoints import SnapshotEndpointNormalizer
    from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotRecord
    from miles.utils.audit_utils.config_snapshot.normalizer import (
        collect_generated_values,
        normalize_record,
        normalized_source_name,
    )
    from miles.utils.audit_utils.process_identity import TrainProcessIdentity

    assert not output.exists()
    assert (
        Path(sys.modules[normalize_record.__module__].__file__).resolve()
        == (source_directory / "miles/utils/audit_utils/config_snapshot/normalizer.py").resolve()
    )
    source_files = [
        source_directory / "miles/utils/audit_utils/config_snapshot" / name
        for name in ("models.py", "normalizer.py", "endpoints.py")
    ]
    records = []
    for path in sorted(raw_directory.rglob("attempt.json")):
        attempt = json.loads(path.read_text())
        files = sorted((path.parent / "records").glob("*.json"))
        row = {
            "attempt": str(path),
            "raw_sha256": {str(item): hashlib.sha256(item.read_bytes()).hexdigest() for item in files},
        }
        if not attempt["completed"]:
            row["error"] = "incomplete-attempt"
            records.append(row)
            continue
        try:
            captures = [ConfigSnapshotRecord.model_validate_json(item.read_text()) for item in files]
            endpoints = SnapshotEndpointNormalizer.create(captures)
            generated_values = collect_generated_values(captures)
            by_capture = defaultdict(list)
            for capture in captures:
                by_capture[capture.context.capture_id].append(capture)
            processes = {}
            for captures in by_capture.values():
                context = captures[0].context
                name = f"{context.name}/{context.deploy_component}/{context.deploy_instance_id}/{normalized_source_name(context.source)}"
                stages = {}
                for capture in captures:
                    assert capture.context == context
                    stage = capture.point.to_key()
                    assert stage not in stages
                    stages[stage] = normalize_record(
                        capture,
                        endpoints=endpoints,
                        generated_values=generated_values[
                            (context.name, context.run_uuid, context.deploy_instance_id)
                        ],
                    )
                assert "process_config-0000" in stages
                rank = [context.source.rank_within_cell] if isinstance(context.source, TrainProcessIdentity) else []
                if name in processes:
                    assert json.dumps(processes[name]["stages"], sort_keys=True) == json.dumps(stages, sort_keys=True)
                    assert not set(processes[name]["ranks"]) & set(rank)
                    processes[name]["ranks"] = sorted(processes[name]["ranks"] + rank)
                else:
                    processes[name] = {"ranks": rank, "stages": stages}
            row["expanded"] = processes
        except Exception as error:
            logger.exception("Historical normalizer replay failed: %s", path)
            row["error"] = f"{type(error).__name__}: {error}"
        records.append(row)
    output.write_text(
        json.dumps(
            {
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
                "attempts": records,
            },
            indent=2,
        )
    )
    if any("error" in row for row in records):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
