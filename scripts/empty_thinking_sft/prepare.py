"""Prepare a controlled empty-thinking-boundary SFT ablation without changing messages."""

import hashlib
import json
import os
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import BinaryIO

from tap import Tap
from transformers import AutoTokenizer

TOKENIZER = None
TEMPLATE_SHA = ''


class Args(Tap):
    mode: str
    base: Path
    model: Path
    source: Path | None = None
    output: Path | None = None
    audit: Path | None = None
    expected_source_sha256: str | None = None
    workers: int = 16


def fixed_template(text: str) -> str:
    old = "+ reasoning_content + '\\n</think>\\n\\n'"
    new = "+ reasoning_content + ('\\n' if reasoning_content else '') + '</think>\\n\\n'"
    assert text.count(old) == 1, 'Unexpected assistant serialization template'
    text = text.replace(old, new)
    old_disabled = "'<think>\\n\\n</think>\\n\\n'"
    assert text.count(old_disabled) == 1, 'Unexpected disabled-thinking generation template'
    return text.replace(old_disabled, "'<think>\\n</think>\\n\\n'")


def assets(args: Args) -> None:
    assert args.base.resolve() != args.model.resolve()
    args.model.mkdir(parents=True, exist_ok=False)
    original = (args.base / 'chat_template.jinja').read_text()
    updated = fixed_template(original)
    for path in args.base.iterdir():
        if path.name == 'chat_template.jinja':
            (args.model / path.name).write_text(updated)
        elif path.name == 'tokenizer_config.json':
            config = json.loads(path.read_text())
            if 'chat_template' in config:
                assert config['chat_template'] == original
                config['chat_template'] = updated
            (args.model / path.name).write_text(json.dumps(config, indent=2)+'\n')
        else:
            os.symlink(path.resolve(), args.model / path.name, target_is_directory=path.is_dir())
    index = json.loads((args.model / 'model.safetensors.index.json').read_text())
    for shard in set(index['weight_map'].values()):
        assert (args.model / shard).resolve() == (args.base / shard).resolve()
        assert (args.model / shard).stat().st_size > 0
    print(json.dumps({'model': str(args.model), 'template_sha256': hashlib.sha256(updated.encode()).hexdigest(),
                      'fresh_weight_shards': len(set(index['weight_map'].values()))}), flush=True)


def initialize(model: str) -> None:
    global TOKENIZER, TEMPLATE_SHA
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)
    TEMPLATE_SHA = hashlib.sha256(Path(model, 'chat_template.jinja').read_bytes()).hexdigest()


def transform(item: tuple[int, bytes]) -> tuple[str, dict]:
    index, raw = item
    row = json.loads(raw)
    metadata = row['metadata']
    old_tokens = metadata['sft_tokens']
    mask = metadata['sft_mask']
    assert len(old_tokens) == len(mask) and any(mask)
    new_tokens = old_tokens.copy()
    position = 0
    assistant_count = 0
    changes = 0
    while True:
        try:
            position = old_tokens.index(248045, position)
        except ValueError:
            break
        if old_tokens[position+1:position+4] == [74455, 198, 248068]:
            assistant_count += 1
            assert mask[position:position+5] == [0]*5
            assert mask[position+5] == 1
            if old_tokens[position+4:position+6] == [271, 248069]:
                new_tokens[position+4] = 198
                changes += 1
            else:
                assert old_tokens[position+4] == 198
        position += 1
    assert assistant_count == sum(m['role'] == 'assistant' for m in row['messages'])
    # The real consumer: re-render and tokenize the COMPLETE conversation with
    # the modified template. This verifies all histories, tools, and BPE seams.
    actual = TOKENIZER.apply_chat_template(row['messages'], tools=row.get('tools'),
        preserve_thinking=True, tokenize=True, return_dict=False)
    assert actual == new_tokens, f'Full-template token mismatch at source row {index}'
    assert sum(a != b for a, b in zip(old_tokens, new_tokens, strict=True)) == changes
    # A 271 -> 198 replacement occupies the same position and length. Keeping
    # the checked mask exactly preserves every supervised target and boundary.
    metadata['sft_tokens'] = actual
    metadata['sft_mask'] = list(mask)
    metadata['empty_thinking_boundary_fix'] = {'version': 1, 'changes': changes,
        'template_sha256': TEMPLATE_SHA, 'messages_unchanged': True, 'mask_unchanged': True}
    result = {'index': index, 'tokens': len(actual), 'supervised': sum(mask),
              'changed_boundaries': changes, 'assistant_turns': assistant_count}
    return json.dumps(row, ensure_ascii=False, separators=(',', ':'))+'\n', result


def dataset(args: Args) -> None:
    assert args.source and args.output and args.audit
    assert args.source.resolve() != args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    source_hash = hashlib.sha256()
    output_hash = hashlib.sha256()
    records = []

    def items(stream: BinaryIO) -> Iterator[tuple[int, bytes]]:
        for index, raw in enumerate(stream):
            source_hash.update(raw)
            yield index, raw

    # map prefetch is bounded on Python 3.12 by explicit chunks of 32 rows.
    with args.source.open('rb') as stream, args.output.open('xb') as output:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(str(args.model),)) as pool:
            iterator = iter(items(stream))
            while True:
                batch = []
                for _ in range(32):
                    item = next(iterator, None)
                    if item is None:
                        break
                    batch.append(item)
                if not batch:
                    break
                for encoded, record in pool.map(transform, batch, chunksize=1):
                    data = encoded.encode()
                    output.write(data)
                    output_hash.update(data)
                    records.append(record)
                    counts['sessions'] += 1
                    for key in ('tokens', 'supervised', 'changed_boundaries', 'assistant_turns'):
                        counts[key] += record[key]
                print(json.dumps({'verified_sessions': counts['sessions'], 'changed_boundaries': counts['changed_boundaries']}), flush=True)
    if args.expected_source_sha256:
        assert source_hash.hexdigest() == args.expected_source_sha256
    report = {'sessions': counts['sessions'], 'total_tokens': counts['tokens'],
        'supervised_tokens': counts['supervised'], 'max_tokens': max(r['tokens'] for r in records),
        'changed_boundaries': counts['changed_boundaries'], 'assistant_turns': counts['assistant_turns'],
        'source_sha256': source_hash.hexdigest(), 'output_sha256': output_hash.hexdigest(),
        'messages_unchanged': True, 'all_masks_unchanged': True, 'all_rows_match_modified_template': True,
        'template_sha256': hashlib.sha256((args.model/'chat_template.jinja').read_bytes()).hexdigest(),
        'rows': records}
    args.audit.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}), flush=True)


def main() -> None:
    args = Args().parse_args()
    if args.mode == 'assets':
        assets(args)
    else:
        assert args.mode == 'dataset'
        dataset(args)


if __name__ == '__main__':
    main()
