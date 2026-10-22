import hashlib
from pathlib import Path


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
