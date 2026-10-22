import json
from pathlib import Path
from typing import Annotated

import typer
from pydantic import JsonValue

from miles.utils.file_utils import atomic_write_text
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison import _differences
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_identity import _snapshot_identity
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_input import _load_json

app = typer.Typer(add_completion=False)


@app.command()
def audit(
    before: Annotated[Path, typer.Option()],
    after: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    assert not output.exists()
    original = expand_snapshot_directory(directory=before)
    actual = expand_snapshot_directory(directory=after)
    cases = []
    for name in sorted(original.keys() & actual.keys()):
        if not name.startswith("argument_schema/"):
            continue
        left, right = original[name], actual[name]
        assert isinstance(left, dict) and isinstance(right, dict)
        old_actions = left["schema"]["parser"]["state"]["_actions"]
        new_actions = right["schema"]["parser"]["state"]["_actions"]
        cases.append(
            {
                "case": name,
                "input_changes": [
                    item.model_dump()
                    for item in _differences(
                        before={key: left[key] for key in ("argv", "legacy")},
                        after={key: right[key] for key in ("argv", "legacy")},
                        path=[],
                    )
                ],
                "before_order": [_action_identity(action=action) for action in old_actions],
                "after_order": [_action_identity(action=action) for action in new_actions],
                "action_changes": [
                    {**item.model_dump(), "evidence": _action_evidence(difference=item)}
                    for item in _differences(
                        before=_actions_by_identity(actions=old_actions),
                        after=_actions_by_identity(actions=new_actions),
                        path=[],
                    )
                ],
            }
        )
    atomic_write_text(
        path=output,
        text=json.dumps(
            {
                "before_inventory": _snapshot_identity(directory=before),
                "after_inventory": _snapshot_identity(directory=after),
                "cases": cases,
            },
            indent=2,
        ),
    )


def _action_evidence(*, difference: SnapshotDifference) -> dict[str, str]:
    destination = _load_json(difference.path[0])[0]
    attribute = difference.path[1] if len(difference.path) > 1 else "action"
    if destination.startswith(("sglang_", "eval_sglang_")):
        return {
            "verdict": "dependency-cli-contract-change",
            "source": "SGLang f7bd7caef8231ea76406f86dedcd9b2b2579cf1a -> 203f4bbc70e54b8e5e8282262cdb88752068a99e; python/sglang/srt/arg_groups/fields and arg_groups/cli.py",
            "reason": "The full action object is captured from the pinned dependency registration, including eval overrides. Changes to defaults, accepted choices, converter callables and option aliases are real CLI behavior changes. A help-only field changes documentation. Neither kind is a mechanical snapshot approval; current runtime dispatch still needs its own captured evidence.",
        }
    if attribute in {"help", "group"}:
        return {
            "verdict": "parser-documentation-change",
            "source": "miles/utils/args/schema.py:BaseConfig.add_arguments; miles/utils/args/configs; miles/utils/arguments.py",
            "reason": "This difference is exactly one help string or display group; parser conversion, default, const and accepted choices are separate action fields in this inventory. Documentation can describe changed behavior, so this metadata-only observation does not certify the described runtime behavior.",
        }
    if (
        destination in {"dsa_impl", "miles_dsa_topk_backend", "score_centering_is"}
        and attribute == "type"
        and difference.before.present
        and difference.before.value is None
        and difference.after.present
        and difference.after.value == {"$callable": "builtins.str"}
    ):
        return {
            "verdict": "implicit-string-conversion-made-explicit",
            "source": "miles/utils/args/schema.py; miles/utils/args/configs/custom_megatron_plugins.py; miles/utils/args/configs/algo.py; argparse.ArgumentParser._get_value",
            "reason": "Argparse's absent type uses its identity converter on CLI strings; explicit builtins.str returns the same strings. The exact before/after type is preserved above; choices and defaults are compared separately.",
        }
    contracts = {
        "ft_components": (
            "miles/utils/arguments.py:_resolve_ft_components; miles/utils/args/configs/fault_tolerance.py",
            "Upstream adds weight_update to accepted FT components and enables it in the default selection. Its backend/p2p/colocate validation is real behavior; FSDP default rejection and explicit rollout-only positives are covered separately.",
        ),
        "no_load_optim": (
            "miles/utils/args/configs/backend_fields.py; miles/backends/fsdp_utils/checkpoint.py",
            "The common backend argument registration now exposes this switch to FSDP as well as Megatron; it controls checkpoint optimizer loading. This is an added supported CLI surface, not an ignored parser field.",
        ),
        "no_load_rng": (
            "miles/utils/args/configs/backend_fields.py; miles/backends/fsdp_utils/checkpoint.py",
            "The common backend argument registration now exposes this switch to FSDP as well as Megatron; it controls checkpoint RNG loading. This is an added supported CLI surface, not an ignored parser field.",
        ),
        "padded_vocab_size": (
            "tests/fast/argument_snapshot/argparser/scenarios.py; miles/utils/args/schema.py; miles/backends/fsdp_utils",
            "The removed FSDP-only fixture option was not consumed by FSDP execution. M30 op129 rejects accepting an argument that cannot work through the full flow. It is now a real CLI rejection, not numerical equivalence.",
        ),
        "custom_agent_function_path": (
            "miles/utils/args/configs/rollout.py; miles/utils/args/custom_function.py",
            "The typed hook registry exposes the custom agent hook in the common parser. None is an unset hook; non-None input selects an actual consumer and requires separate execution evidence.",
        ),
        "trainer_init_expected_num_cells": (
            "miles/utils/args/configs/cluster.py; miles/ray/specs/train.py",
            "This explicit startup-cell count is a new interface. Omitted input derives the previous initial count, tested by the before-runtime replay; explicit input changes startup gating and is not approved as equal.",
        ),
        "snapshot_hook": (
            "tests/fast/argument_snapshot/argparser/scenarios.py:_LegacyHook; miles/utils/args/custom_function.py:add_user_provided_function_arguments",
            "Only legacy rollout/generate previously skipped add_arguments; the adapter now registers their declared snapshot_hook=23. The inference variant already registered it. The fixture callable rename and the registration capability change are reported separately.",
        ),
    }
    if destination.startswith("tinker_"):
        return {
            "verdict": "added-cli-interface",
            "source": "miles/utils/args/configs/tinker.py; miles/tinker",
            "reason": "The typed common configuration exposes the gateway's host/port/model/checkpoint settings. These exact values are new parser declarations; ordinary training does not start the gateway from their mere presence. Gateway startup remains a distinct entrypoint.",
        }
    if destination in contracts:
        source, reason = contracts[destination]
        return {"verdict": "documented-cli-behavior-change", "source": source, "reason": reason}
    return {
        "verdict": "unverified-parser-change",
        "source": "",
        "reason": "No value-specific source rationale is registered; retain strict failure.",
    }


def _actions_by_identity(*, actions: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    result = {}
    for action in actions:
        key = _action_identity(action=action)
        assert key not in result, f"Duplicate parser action identity: {key}"
        result[key] = action
    return result


def _action_identity(*, action: dict[str, JsonValue]) -> str:
    # Positive/negative switches can share a destination. Keeping their option
    # strings in the identity prevents one action from hiding the other.
    return json.dumps([action["dest"], action["option_strings"]])


if __name__ == "__main__":
    app()
