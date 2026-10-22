from pathlib import Path
from typing import Annotated

import typer
import yaml

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_input import _load_json

app = typer.Typer(add_completion=False)


@app.command()
def prepare(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    assert not output.exists()
    original = _load_json(before.read_text())
    actual = _load_json(after.read_text())
    assert original["legacy"] is True and actual["legacy"] is False
    assert original["scenario_argv"] == actual["scenario_argv"]
    assert original["worker_argv"] == actual["worker_argv"]
    assert original["workers"].keys() == actual["workers"].keys()
    identities = {
        "rollout-executor": "rollout_executor",
        "inference-controller": "inference_controller",
        "trainer-controller-actor": "trainer_controller_actor",
        "trainer-engine-actor": "actor_cell00000",
    }
    assert original["workers"].keys() == identities.keys()
    for name in identities:
        assert original["workers"][name]["worker_class"] == actual["workers"][name]["worker_class"]
    for side, record in (("before", original), ("after", actual)):
        directory = output / side
        (directory / "parsed_config").mkdir(parents=True)
        (directory / "runtime_config").mkdir()
        (directory / "parsed_config/chart_fixture.yaml").write_text(yaml.safe_dump({"config": record["parsed"]}))
        processes = {
            f"chart-fixture/run-0000/all/default/{identity}": {
                "ranks": [0] if identity == "actor_cell00000" else [],
                "base": {"args": record["workers"][name]["args"]},
                "diffs": {},
            }
            for name, identity in identities.items()
        }
        # This is a synthetic shape for applying the exact same audited field
        # comparison to CPU spec observations, not a claim that workers ran.
        (directory / "runtime_config/chart_fixture.yaml").write_text(yaml.safe_dump({"processes": processes}))


if __name__ == "__main__":
    app()
