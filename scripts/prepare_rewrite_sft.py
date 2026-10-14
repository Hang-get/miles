"""Freeze accepted rewritten sessions and audit their actual training tokens."""

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
    roots: list[str]
    tokenizer: str
    output: str
    workers: int = 8


def initialize(path: str) -> None:
    global TOKENIZER, MASKER
    TOKENIZER = AutoTokenizer.from_pretrained(path)
    MASKER = QwenSftLossMask(TOKENIZER, {"preserve_thinking": True})


def prepare(path: str) -> tuple[dict, dict]:
    row = json.loads(Path(path).read_text())
    tokens, mask = MASKER.token_ids_and_mask(row["messages"], row.get("tools"))
    reference = TOKENIZER.apply_chat_template(row["messages"], tools=row.get("tools"),
                                             preserve_thinking=True, tokenize=True, return_dict=False)
    if tokens != reference or len(mask) != len(tokens) or not any(mask) or len(tokens) > 262144:
        raise ValueError(f"SFT audit failed: {path}")
    metadata = dict(row.get("metadata") or {})
    metadata.update(sft_tokens=tokens, sft_mask=mask, source_artifact=path)
    row["metadata"] = metadata
    return row, {"artifact": path, "tokens": len(tokens), "supervised": sum(mask)}


def main(args: Args) -> None:
    paths = [str(p.parent / "trainable.json") for root in args.roots
             for p in sorted(Path(root).glob("sessions/*/result.json"))
             if json.loads(p.read_text()).get("training_ready")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    identities = set()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                             initargs=(args.tokenizer,)) as pool, (out / "sft_train.jsonl").open("w") as stream:
        for row, record in pool.map(prepare, paths):
            identity_messages = [{k: v for k, v in m.items() if k != "reasoning_content"}
                                 for m in row["messages"]]
            identity = hashlib.sha256(json.dumps([identity_messages, row.get("tools")],
                                      sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if identity in identities:
                raise ValueError("Duplicate non-reasoning conversation in combined dataset")
            identities.add(identity)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            records.append(record)
            if len(records) % 100 == 0:
                print(json.dumps({"audited": len(records), "of": len(paths)}), flush=True)
    with (out / "sft_train.jsonl").open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = {"sessions": len(records), "total_tokens": sum(r["tokens"] for r in records),
             "trainable_tokens": sum(r["supervised"] for r in records),
             "max_tokens": max(r["tokens"] for r in records), "data_sha256": digest,
             "mask": "QwenSftLossMask grouped-tool assistant-only; preserve_thinking=True",
             "records": records}
    (out / "mask-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    main(Args().parse_args())
