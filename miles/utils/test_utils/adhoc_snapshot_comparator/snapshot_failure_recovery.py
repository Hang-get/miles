import hashlib
import json
import re
from pathlib import Path
from typing import Annotated

import typer

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory

app = typer.Typer(add_completion=False)


@app.command()
def audit(
    log: Annotated[Path, typer.Option()],
    expected: Annotated[Path, typer.Option()],
    output_directory: Annotated[Path, typer.Option()],
) -> None:
    assert not output_directory.exists()
    snapshots = _recover_actual_blocks(text=log.read_text())
    actual_directory = output_directory / "actual"
    for relative, text in snapshots.items():
        target = actual_directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    original = expand_snapshot_directory(directory=expected)
    actual = expand_snapshot_directory(directory=actual_directory)
    differences = []
    for name, value in actual.items():
        assert name in original, f"No recorded counterpart for recovered test output: {name}"
        differences.extend(_differences(before=original[name], after=value, path=[name]))
    result = {
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "recovered_files": list(snapshots),
        "expanded_cases": sorted(actual),
        "differences": [item.model_dump() for item in differences],
        "scope": "Only actual YAML blocks already emitted by the failed read-only tests are recovered. Passing assertions emit no block. This does not update any golden or collect a new production snapshot.",
    }
    (output_directory / "comparison.json").write_text(json.dumps(result, indent=2))
    if differences:
        raise typer.Exit(code=1)


def _recover_actual_blocks(*, text: str) -> dict[str, str]:
    result = {}
    current_path = None
    active = None
    prefix = ""
    lines = []
    for line in text.splitlines():
        if match := re.match(r"^E\s+--- /root/miles/tests/snapshots/(.+)$", line):
            current_path = match[1]
        if match := re.match(r"^(E\s+)--- BEGIN ACTUAL (.+) ---$", line):
            assert active is None and current_path is not None
            assert Path(current_path).name == match[2]
            assert not Path(current_path).is_absolute() and ".." not in Path(current_path).parts
            active, prefix, lines = current_path, match[1], []
        elif active is not None and line.strip().endswith("--- END ACTUAL ---"):
            value = "\n".join(lines) + "\n"
            assert active not in result or result[active] == value, f"Conflicting actual outputs: {active}"
            result[active] = value
            active = None
        elif active is not None:
            assert line.startswith(prefix) or line.strip() == "E", f"Unexpected traceback prefix: {line!r}"
            lines.append(line[len(prefix) :] if line.startswith(prefix) else "")
    assert active is None, "Truncated actual YAML block"
    assert result, "No complete actual YAML blocks"
    return result


if __name__ == "__main__":
    app()
