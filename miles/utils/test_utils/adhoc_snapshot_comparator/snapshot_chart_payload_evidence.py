import hashlib
import json
import shlex
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
import yaml
from pydantic import JsonValue

app = typer.Typer(add_completion=False)


@app.command()
def compare(
    snapshots: Annotated[Path, typer.Option()],
    observed: Annotated[Path, typer.Option()],
    before_observed: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    assert not output.exists()
    from miles.ray.specs.entrypoint import SERVE_SPEC_CLASSES
    from miles.utils.test_utils.snapshot import snapshot_values
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences
    from miles.utils.workers.serving.utils import parse_serve_worker_config

    observation = json.loads(observed.read_text())
    original = json.loads(before_observed.read_text())
    assert original["legacy"] is True and observation["legacy"] is False
    records = []
    for source in sorted(snapshots.glob("*.yaml")):
        for path, payload in _payloads(value=list(yaml.safe_load_all(source.read_text())), path=[source.name]):
            worker = parse_serve_worker_config(payload)
            spec_class = SERVE_SPEC_CLASSES[worker.worker_type]
            config = spec_class.config_class.model_validate(worker.args)
            spec = spec_class.create(config)
            expected = observation["workers"][spec.name]["args"]
            actual = snapshot_values(config)
            differences = _differences(before=expected, after=actual, path=[])
            records.append(
                {
                    "path": path,
                    "worker": spec.name,
                    "differences": [x.model_dump() for x in differences],
                    "evidence": [
                        _explain_pair(pair=x.model_dump(), original=original, actual=actual) for x in differences
                    ],
                }
            )
    assert len(records) == 8, len(records)
    output.write_text(
        json.dumps(
            {
                "observed_sha256": hashlib.sha256(observed.read_bytes()).hexdigest(),
                "before_observed_sha256": hashlib.sha256(before_observed.read_bytes()).hexdigest(),
                "comparison_scope": "Current payloads decoded by the current production schema, compared with the explicitly supplied historical CPU spec observation; not a new production capture.",
                "records": records,
            },
            indent=2,
        )
    )
    if any(record["differences"] for record in records):
        raise typer.Exit(code=1)


def _explain_pair(*, pair: dict, original: dict, actual: dict | None = None) -> dict[str, JsonValue]:
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_case_evidence import _case_evidence
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_dependency_evidence import _dependency_evidence

    if (evidence := _dependency_evidence(difference=SnapshotDifference.model_validate(pair))) is not None:
        return evidence
    if actual is not None:
        case = "parsed_config/decoded-chart-payload"
        difference = SnapshotDifference.model_validate({**pair, "path": [case, "config", *pair["path"]]})
        if (evidence := _case_evidence(difference=difference, actual={case: {"config": actual}})) is not None:
            evidence["observation_scope"] = (
                "Actual decoded chart payload; synthetic lookup key, not a runtime capture."
            )
            return evidence
    path, before, after = pair["path"], pair["before"], pair["after"]
    if (
        path in (["extra_high_precision_layers_hf"], ["extra_high_precision_layers_megatron"])
        and before == {"present": True, "value": {"$tuple": []}}
        and after == {"present": True, "value": []}
    ):
        # The schema accepts both sequence types; all consumers test emptiness,
        # iterate, or explicitly create a tuple. No consumer mutates this input.
        return {
            "verdict": "empty-sequence-consumer-preserved",
            "source": "miles/utils/args/configs/train.py:TrainConfig; miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py; quantizer_nvfp4.py; tools/convert_hf_to_mxfp8.py; tools/convert_hf_to_nvfp4.py",
            "reason": "The serialized empty list and parser empty tuple take the identical empty-quantization-exclusion branch. Their types remain strictly different.",
        }
    if (
        len(path) == 2
        and path[0] == "router_args"
        and path[1] in {"allow_requests_without_routing_key", "json_log", "test_external_dp_routing"}
        and before == {"present": True, "value": False}
        and type(before["value"]) is bool
        and after == {"present": False, "value": None}
    ):
        old_field = "router_" + path[1]
        if old_field in original["parsed"] and original["parsed"][old_field] is False:
            # Both pinned source revisions produce False in this replay
            # environment; the historical payload omits the field entirely.
            return {
                "verdict": "historical-replay-environment-difference",
                "witness": {
                    "before_parser_field": old_field,
                    "before_parser_value": False,
                    "after_parser_value": False,
                    "historical_payload_present": False,
                },
                "reason": "The same replay environment produces the exact False on both source revisions. Historical capture lacks the field, so this is not a demonstrated before/after code regression. Do not fill the historical field or claim a byte-identical historical replay.",
            }
    return {
        "verdict": "requires-specific-witness",
        "reason": "This exact decoded payload difference has no declared consumer or same-environment witness.",
    }


def _payloads(*, value: JsonValue, path: list[str]) -> Iterator[tuple[list[str], str]]:
    if isinstance(value, dict):
        for name, child in value.items():
            yield from _payloads(value=child, path=[*path, name])
    elif isinstance(value, list):
        if "--config" in value and "miles.utils.workers.serving.serve" in value:
            payload = value[value.index("--config") + 1]
            assert isinstance(payload, str)
            yield path, payload
        else:
            for index, child in enumerate(value):
                yield from _payloads(value=child, path=[*path, str(index)])
    elif isinstance(value, str) and "--config " in value:
        tokens = shlex.split(value)
        if "--config" in tokens and "miles.utils.workers.serving.serve" in tokens:
            yield path, tokens[tokens.index("--config") + 1]


if __name__ == "__main__":
    app()
