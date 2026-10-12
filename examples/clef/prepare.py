"""Materialize and audit schema records without loading model weights or GPUs."""

import json
import random
from collections import Counter
from pathlib import Path

from tap import Tap
from transformers import AutoProcessor

from examples.clef.data import augment_example, encode_example, file_sha256, read_examples


class Args(Tap):
    input_dir: str
    output_dir: str
    model_dir: str
    max_length: int = 65536
    seed: int = 261003


def main() -> None:
    args = Args(underscores_to_dashes=True).parse_args()
    source, output = Path(args.input_dir), Path(args.output_dir)
    train, validation = read_examples(source / "train.jsonl"), read_examples(source / "validation.jsonl")
    if {e.record["id"] for e in train} & {e.record["id"] for e in validation}:
        raise ValueError("train/validation IDs overlap")
    tokenizer = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True).tokenizer
    lengths = []
    for index, example in enumerate(train + validation):
        lengths.append(len(encode_example(tokenizer, example, args.max_length).encoded.input_ids))
        # Exercise the larger augmented schema on every record during preflight.
        augmented = augment_example(example, random.Random(args.seed + index), multi_field_fraction=1)
        lengths.append(len(encode_example(tokenizer, augmented, args.max_length).encoded.input_ids))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"source_dir": str(source), "max_length": args.max_length, "maximum_encoded_tokens": max(lengths), "splits": {}}
    for name, examples in (("train", train), ("validation", validation)):
        path = output / f"{name}.jsonl"
        with path.open("w") as writer:
            for example in examples:
                writer.write(json.dumps({"record": example.record, "targets": example.targets, "source": example.source}, ensure_ascii=False) + "\n")
        roundtrip = read_examples(path)
        if roundtrip != examples:
            raise ValueError(f"prepared-data roundtrip mismatch: {name}")
        manifest["splits"][name] = {"questions": len(examples), "sources": dict(Counter(e.source for e in examples)),
                                    "sha256": file_sha256(path), "source_sha256": file_sha256(source / f"{name}.jsonl")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("DATA_PREPARED", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
