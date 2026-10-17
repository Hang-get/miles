"""Audit a JSONL corpus against the exact thinking-preserving training mask."""

import gzip
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tap import Tap
from transformers import AutoTokenizer

from experiments.shi.data_dashboard.viewer.sft_loss_mask import QwenSftLossMask

TOKENIZER: Any = None
MASKER: Any = None


class Args(Tap):
    source: str
    tokenizer: str
    output: str
    expected_sessions: int
    workers: int = 32


def initialize(path: str) -> None:
    global TOKENIZER, MASKER
    TOKENIZER = AutoTokenizer.from_pretrained(path)
    MASKER = QwenSftLossMask(TOKENIZER, {"preserve_thinking": True})


def prepare(line: str) -> tuple[dict, dict]:
    row = json.loads(line)
    tokens, mask = MASKER.token_ids_and_mask(row["messages"], row.get("tools"))
    reference = TOKENIZER.apply_chat_template(
        row["messages"], tools=row.get("tools"), preserve_thinking=True,
        tokenize=True, return_dict=False,
    )
    if tokens != reference or len(mask) != len(tokens) or not any(mask) or len(tokens) > 262144:
        raise ValueError("Training template/mask validation failed")
    messages = [{k: v for k, v in m.items() if k != "reasoning_content"} for m in row["messages"]]
    identity = hashlib.sha256(json.dumps([messages, row.get("tools")], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    row["metadata"] = {**(row.get("metadata") or {}), "sft_tokens": tokens, "sft_mask": mask}
    return row, {"identity": identity, "tokens": len(tokens), "supervised": sum(mask)}


def main(args: Args) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "sft_train.jsonl"
    if target.exists():
        raise FileExistsError(target)
    records = []
    identities = set()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(args.tokenizer,)) as pool:
        with gzip.open(args.source, "rt") as source, target.open("w") as stream:
            while True:
                batch = [source.readline() for _ in range(args.workers * 2)]
                batch = [line for line in batch if line]
                if not batch:
                    break
                for row, record in pool.map(prepare, batch):
                    if record["identity"] in identities:
                        raise ValueError("Duplicate non-reasoning conversation")
                    identities.add(record["identity"])
                    records.append(record)
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(json.dumps({"audited": len(records), "expected": args.expected_sessions}), flush=True)
    assert len(records) == args.expected_sessions
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = {"sessions": len(records), "total_tokens": sum(r["tokens"] for r in records),
             "trainable_tokens": sum(r["supervised"] for r in records),
             "max_tokens": max(r["tokens"] for r in records), "data_sha256": digest,
             "preserve_thinking": True, "records": records}
    (output / "mask-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main(Args().parse_args())
