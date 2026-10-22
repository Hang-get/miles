import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory

app = typer.Typer(add_completion=False)


@app.command()
def audit(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    assert not output.exists(), f"Report already exists: {output}"
    original = expand_snapshot_directory(directory=before)
    actual = expand_snapshot_directory(directory=after)
    records = _unpaired_cases(before=original, after=actual)
    atomic_write_text(path=output, text=json.dumps({"unpaired_cases": records}, indent=2))
    if records:
        raise typer.Exit(code=1)


def _unpaired_cases(*, before: dict[str, JsonValue], after: dict[str, JsonValue]) -> list[dict]:
    records = []
    for case in sorted(before.keys() ^ after.keys()):
        observed_side = "before" if case in before else "after"
        observed = before[case] if case in before else after[case]
        required = []
        if case.startswith("runtime_config/"):
            assert isinstance(observed, dict)
            for process, value in observed.items():
                assert isinstance(value, dict) and isinstance(stages := value["stages"], dict)
                required.append(
                    {
                        "process": process,
                        "ranks": value["ranks"],
                        "stages": list(stages),
                        "inputs": _recorded_inputs(stages=stages),
                    }
                )
        # This is the observed counterpart's coverage requirement, not a
        # fabricated missing capture. A new run must preserve its inputs and
        # generation boundaries; another case or topology cannot fill the gap.
        records.append(
            {
                "case": case,
                "observed_side": observed_side,
                "missing_side": "after" if observed_side == "before" else "before",
                "required_counterpart": required,
                "verdict": "missing-paired-capture",
                "next_check": (
                    "Recover an actual capture with the same case, run inputs, process owners and stages. "
                    "If source or inputs no longer exist, retain this as historical inventory and list a new "
                    "paired scenario separately; do not rename another case or synthesize values."
                ),
            }
        )
    return records


def _recorded_inputs(*, stages: dict[str, JsonValue]) -> dict[str, JsonValue]:
    initial = stages.get("process_config-0000")
    if not isinstance(initial, dict) or not isinstance(args := initial.get("args"), dict):
        return {}
    # Keep the complete observed input: selecting familiar launch fields loses
    # topology, custom hooks and backend-specific settings needed for recapture.
    # This remains an after-side requirement, never a synthesized before value.
    return args


if __name__ == "__main__":
    app()
