import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any

import pytest
import typer

app = typer.Typer(add_completion=False)


@app.command()
def capture(
    output: Annotated[Path, typer.Option()],
    legacy: Annotated[bool, typer.Option()] = False,
) -> None:
    assert not output.exists()
    from tests.fast.charts.miles_run import test_snapshot as chart

    from miles.utils.test_utils.snapshot import snapshot_values

    # Run this same observer with PYTHONPATH pointing at each pinned source tree.
    # The real fixture parser builds specs; no worker is instantiated or launched.
    # Old ctor closures hold the actual Namespace hidden by incomplete WORKER_ARGV.
    with pytest.MonkeyPatch.context() as monkeypatch:
        with contextmanager(chart.parser_process_env.__wrapped__)(monkeypatch):
            with chart.override_argv(chart.SCENARIO_ARGV):
                args = chart.parse_args()
            specs = chart.compute_specs(args)
            records = {}
            for spec in specs:
                if legacy:
                    from miles.utils.workers.worker_spec import ServeWorkerSpec

                    if not isinstance(spec, ServeWorkerSpec):
                        continue
                    closure = inspect.getclosurevars(spec.ctor_kwargs).nonlocals
                    assert "args" in closure, (spec.name, closure.keys())
                    config = closure["args"]
                else:
                    from miles.utils.workers.worker_spec import BaseServeSpec

                    if not isinstance(spec, BaseServeSpec):
                        continue
                    config = spec.args
                records[spec.name] = {"worker_class": spec.worker_class, "args": snapshot_values(config)}
            result: dict[str, Any] = {
                "legacy": legacy,
                "scenario_argv": chart.SCENARIO_ARGV,
                "worker_argv": chart.WORKER_ARGV,
                "parsed": snapshot_values(args),
                "workers": records,
            }
            text = json.dumps(result, indent=2).replace(str(chart.FIXTURE_DIR), "<FIXTURE_DIR>")
    output.write_text(text)


if __name__ == "__main__":
    app()
