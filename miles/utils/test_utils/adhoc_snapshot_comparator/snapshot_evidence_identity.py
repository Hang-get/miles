import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

from pydantic import JsonValue


def _snapshot_identity(*, directory: Path) -> dict[str, str | int]:
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    assert files, f"No snapshot files: {directory}"
    digest = hashlib.sha256()
    for path in files:
        name = path.relative_to(directory).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    # A directory fingerprint identifies the actual compared inventory. It is
    # not a claim that these mixed historical captures ran on the current HEAD.
    return {"directory": str(directory.resolve()), "files": len(files), "sha256": digest.hexdigest()}


def _value_sha256(value: JsonValue) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _implementation_identity() -> dict[str, JsonValue]:
    root = Path(__file__).resolve().parents[3]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        name, data = path.relative_to(root).as_posix().encode(), path.read_bytes()
        digest.update(len(name).to_bytes(8, "big") + name + len(data).to_bytes(8, "big") + data)
    dependencies = {}
    for name in ("pydantic", "PyYAML", "sglang", "sglang-router"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = "not-installed"
    return {"miles_python_sha256": digest.hexdigest(), "python": sys.version, "dependencies": dependencies}


def _comparison_identity(
    *, before: Path, after: Path, rules: Path, actual: dict[str, JsonValue]
) -> dict[str, JsonValue]:
    return {
        "before": _snapshot_identity(directory=before),
        "after": _snapshot_identity(directory=after),
        "rules": {"path": str(rules.resolve()), "sha256": hashlib.sha256(rules.read_bytes()).hexdigest()},
        "implementation": _implementation_identity(),
        "actual_sha256": _value_sha256(actual),
    }


def _validate_comparison_identity(
    *, identity: dict[str, JsonValue] | None, actual: dict[str, JsonValue] | None
) -> None:
    assert identity is not None, "Comparison has no bound input identity; regenerate it"
    for side in ("before", "after"):
        saved = identity[side]
        assert isinstance(saved, dict)
        assert saved == _snapshot_identity(directory=Path(saved["directory"])), f"Changed {side} inventory"
    rules = identity["rules"]
    assert isinstance(rules, dict)
    assert rules["sha256"] == hashlib.sha256(Path(rules["path"]).read_bytes()).hexdigest(), "Changed comparison rules"
    assert (
        identity["implementation"] == _implementation_identity()
    ), "Changed comparison implementation or dependencies"
    if actual is not None:
        assert identity["actual_sha256"] == _value_sha256(actual), "Changed audit view"
