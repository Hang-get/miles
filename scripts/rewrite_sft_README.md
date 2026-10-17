# Audited conversation SFT helpers

`prepare_rewrite_sft.py` reads a compressed JSONL corpus, checks the actual
token IDs against `preserve_thinking=True`, detects duplicate conversations,
and freezes assistant-only loss masks into row metadata. It requires the
separately versioned `QwenSftLossMask` implementation from the data tooling.
Processing is bounded in small batches to avoid queueing the entire corpus.

`rewrite_sft_rollout.py` consumes those exact arrays without retokenizing.
It retains a final partial batch; use dynamic global batch sizing and
`ceil(sessions / batch_size)` synchronous iterations per epoch.
No reasoning truncation, MTP objective, or new masking policy is introduced.
