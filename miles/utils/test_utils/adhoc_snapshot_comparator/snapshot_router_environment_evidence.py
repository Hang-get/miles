import argparse
import hashlib
import inspect
import json
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import typer
from sglang_router.launch_router import RouterArgs

from miles.utils.test_utils.snapshot import snapshot_values
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_parser_evidence import _actions_by_identity

app = typer.Typer(add_completion=False)


@app.command()
def audit(after: Annotated[Path, typer.Option()], output: Annotated[Path, typer.Option()]) -> None:
    assert not output.exists()
    parser = argparse.ArgumentParser(add_help=False)
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    live = snapshot_values(parser)["state"]["_actions"]
    captured = expand_snapshot_directory(directory=after)["argument_schema/megatron.yaml"]["schema"]["parser"][
        "state"
    ]["_actions"]
    captured_router = [
        action for action in captured if action["dest"].startswith("router_") or action["dest"] == "worker_urls"
    ]
    differences = _differences(
        before=_actions_by_identity(actions=captured_router), after=_actions_by_identity(actions=live), path=[]
    )
    source = Path(inspect.getfile(RouterArgs))
    result = {
        "after_inventory": _snapshot_identity(directory=after),
        "installed_distribution": "sglang-router",
        "installed_version": version("sglang-router"),
        "installed_source": str(source),
        "installed_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "capture_distribution_version": None,
        "capture_version_evidence": "The snapshot records parser behavior but does not identify the independently installed router wheel; a source or SGLang pin is not a substitute for that wheel identity.",
        "captured_order": list(_actions_by_identity(actions=captured_router)),
        "installed_order": list(_actions_by_identity(actions=live)),
        "differences": [item.model_dump() for item in differences],
        "source": "miles/utils/args/configs/router.py:RouterConfig.add_arguments; requirements.txt:sglang-router>=0.2.3; docker/Dockerfile:Install sgl-model-gateway",
        "verdict": "router-environment-contract-mismatch" if differences else "router-declarations-match",
        "repair": "Pin and record the router wheel/source identity together with the capture environment. Reproduce the baseline in that environment or explicitly recollect the affected snapshots after accepting the new CLI contract. Never erase the extra fields or choices in the comparator.",
    }
    output.write_text(json.dumps(result, indent=2))
    if differences:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
