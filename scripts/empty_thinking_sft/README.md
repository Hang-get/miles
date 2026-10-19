# Empty-thinking boundary SFT ablation

These utilities prepare an offline SFT dataset whose assistant messages use
`<think>\n</think>` rather than `<think>\n\n</think>` when reasoning is empty.
Non-empty reasoning, message contents, code, tool calls, and blank lines elsewhere
are preserved. Normal thinking-enabled generation prefixes are unchanged.

`prepare.py --mode assets --base BASE --model MODEL` creates a separate model
asset directory with the adjusted template and references to the original weights.
It never modifies the base checkpoint.

`prepare.py --mode dataset --base BASE --model MODEL --source SOURCE --output OUTPUT
--audit AUDIT` transforms previously audited `metadata.sft_tokens` and verifies every
complete conversation through the actual modified chat template and tokenizer.
The one-token replacement preserves sequence lengths and all loss-mask positions;
the closing-thinking token remains supervised. `--expected_source_sha256` can pin
the original data. Outputs must not already exist.

`launch.py --config CONFIG --preflight` validates the model, epoch, optimizer,
dataset audit, and clean source commits before launch. The manifest is a local
operational artifact, not a place to store credentials. `rollout.py` consumes the
validated frozen tokens and retains the final partial batch.

Use the same runtime dependencies as Miles, plus `transformers` and
`typed-argument-parser` (already provided by the training environment).
