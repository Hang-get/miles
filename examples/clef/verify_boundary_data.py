"""Validate boundary corpus labels against rendered inputs and native encoding."""

import collections
import hashlib
import json
from pathlib import Path

from tap import Tap
from transformers import AutoTokenizer

from examples.clef.data import encode_example, read_examples


class Args(Tap):
    data_dir: str
    tokenizer_dir: str
    benchmark_dir: str | None = None
    prior_data_dir: str | None = None
    max_length: int = 65536


def fingerprint(record: dict) -> str:
    value = {"state": record["state"], "questions": record["questions"]}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_rendered_tools(row: dict) -> None:
    record = row["record"]
    state = record["state"]
    oracle = row["metadata"]["oracle"]
    request = state["conversation"][-1]["content"] if "conversation" in state else record["questions"]["next_action"]["instructions"].split("User request: ", 1)[1]
    capable = [t for t in state["tools"] if t["description"].startswith(f"Supports {oracle['operation']} for records in {oracle['region']}.")]
    supplied = {k: v for k, v in oracle["values"].items() if f"{k} is {v}" in request}
    evidence = state.get("prior_result")
    if evidence:
        assert evidence["request_parameters"] == oracle["values"]
        assert evidence["region"] == oracle["region"]
        assert set(supplied) == set(oracle["values"])
        action = "direct"
    elif not capable:
        action = "cannot_answer"
    elif not set(capable[0]["parameters"]["required"]) <= supplied.keys():
        action = "request_for_info"
    else:
        action = "tool_call"
    assert action == row["metadata"]["mode"]
    target = row["targets"]["next_action"]
    selected = next(k for k, p in target.items() if p == 1)
    assert oracle["action_by_option"][selected] == action
    if action == "tool_call":
        call = json.loads(record["questions"]["next_action"]["criteria"][selected])
        assert call["name"] == capable[0]["name"]
        assert all(call["arguments"][k] == v for k, v in supplied.items())


def main(args: Args) -> None:
    root = Path(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    seen = set()
    families = {}
    report = {"splits": {}, "rendered_tool_oracle": "PASS"}
    for split in ("train", "validation"):
        examples = read_examples(root / f"{split}.jsonl")
        longest = 0
        fields = 0
        for index, example in enumerate(examples):
            signature = fingerprint(example.record)
            assert signature not in seen
            seen.add(signature)
            encoded = encode_example(tokenizer, example, args.max_length)
            longest = max(longest, len(encoded.encoded.input_ids))
            fields += len(encoded.targets)
            if index % 2048 == 0:
                print("ENCODING", split, index, flush=True)
        rows = [json.loads(line) for line in (root / f"{split}.jsonl").open()]
        families[split] = {r["metadata"]["family_id"] for r in rows}
        for row in rows:
            if row["source"] == "boundary_tool_use":
                verify_rendered_tools(row)
        bins = collections.Counter(row["source"] for row in rows)
        report["splits"][split] = {"rows": len(examples), "fields": fields, "max_tokens": longest, "sources": dict(bins)}
    assert not families["train"] & families["validation"]
    for directory, filenames, kind in (
        (args.prior_data_dir, ("train.jsonl", "validation.jsonl"), "prior_dataset"),
        (args.benchmark_dir, ("selected-rows.jsonl", "added-rows.jsonl"), "public_benchmark"),
    ):
        if directory is None:
            continue
        checked = 0
        for name in filenames:
            for line in (Path(directory) / name).open():
                row = json.loads(line)
                record = row.get("record", row)
                assert fingerprint(record) not in seen, (kind, record.get("id"))
                checked += 1
        report[kind] = {"checked": checked, "exact_input_overlap": 0}
    report["family_overlap"] = 0
    report["status"] = "PASS"
    report["limitations"] = "Oracle and native-schema checks are not an independent linguistic audit. No claim of model improvement."
    (root / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main(Args(underscores_to_dashes=True).parse_args())
