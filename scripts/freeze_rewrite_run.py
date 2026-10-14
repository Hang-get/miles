"""Record source revisions and installed packages without capturing credentials."""

import hashlib
import importlib.metadata
import json
import socket
import subprocess
from pathlib import Path

from tap import Tap


class Args(Tap):
    config: str


def main(args: Args) -> None:
    path = Path(args.config)
    config = json.loads(path.read_text())
    repos = [config["miles_path"], "/root/Megatron-LM", "/sgl-workspace/sglang",
             str(path.parent / "src/mbridge")]
    revisions = {}
    for repo in repos:
        dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).strip()
        if dirty:
            raise ValueError(f"Dirty source: {repo}")
        revisions[repo] = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
    config["source_commits"] = revisions
    path.write_text(json.dumps(config, indent=2) + "\n")
    audit = json.loads((path.parent / "data/mask-audit.json").read_text())
    manifest = {"host": socket.gethostname(), "configuration": config,
                "packages": {name: importlib.metadata.version(name) for name in
                             ("torch", "transformers", "ray", "sglang", "mbridge", "megatron-core")},
                "data": {k: v for k, v in audit.items() if k != "records"},
                "mask_source_sha256": hashlib.sha256(Path(
                    "/scratch/17832556/pilot/code/experiments/shi/data_dashboard/viewer/sft_loss_mask.py"
                ).read_bytes()).hexdigest(),
                "launch_command": "env PATH=/opt/sglang/bin:$PATH PYTHONPATH=/scratch/261013-abac3131/src/miles:/root/Megatron-LM:/scratch/17832556/pilot/code /opt/sglang/bin/python /scratch/261013-abac3131/src/miles/scripts/run_rewrite_sft.py --config /scratch/261013-abac3131/config.json",
                "notes": "Fresh HF weights, no optimizer restore; exact two epochs with two partial batches."}
    (path.parent / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    freeze = subprocess.check_output(["/opt/sglang/bin/uv", "pip", "freeze", "--python", "/opt/sglang/bin/python"], text=True)
    (path.parent / "requirements.freeze.txt").write_text(freeze)
    print(json.dumps({"revisions": revisions, "packages": manifest["packages"], "data": manifest["data"]}))


if __name__ == "__main__":
    main(Args().parse_args())
