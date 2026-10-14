# Rewritten-conversation SFT launcher

`prepare_rewrite_sft.py` freezes accepted sessions and checks every token sequence
against the complete thinking-preserving Qwen chat template. Its grouped-tool
loss-mask implementation is imported from the separately snapshotted data tooling.
It stores audited tokens and masks in each row's metadata, so training consumes
exactly the same arrays rather than rebuilding a different mask.

`rewrite_sft_rollout.py` retains the final partial batch instead of dropping sessions
or drawing examples from the next epoch. Enable dynamic global batch size; the last
batch has two examples and the others have four for a 3,870-conversation corpus.

`run_rewrite_sft.py --config <absolute-path> --preflight` validates a frozen run
configuration and clean source revisions. Remove `--preflight` to submit to an
already-started Ray head. Two epochs require 1,936 updates; checkpoint cadence is
968 updates, producing zero-based iterations 967 and 1935.

No generation, evaluation, MTP objective, or change to assistant-token supervision
is introduced by these launch utilities. Data arrays and run artifacts are not
committed to this public repository.
