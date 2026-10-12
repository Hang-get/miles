# Agent rollouts through the official Tinker proxy

This is an independently installable client package. It uses the pinned
`tinker_cookbook.capture.proxy` for OpenAI/Anthropic chat, rendering, tools,
stop conditions and SSE. It imports no Miles code and works with a sampler
from either the Miles gateway or the official Tinker service.

```bash
uv pip install ./examples/multi_lora/tinker_agent
```

Use `RecordingSampler(sampling_client)` with `serve_proxy(renderer, recorder,
model_label=..., default_max_tokens=...)`. Register each trial with
`async with recorder.episode(traj_idx) as episode` and give the agent the
proxy URL with `/r/traj/<traj_idx>` before `/v1/chat/completions` or
`/v1/messages`. One proxy binds one sampler for its entire rollout group.
For sandbox callers, set `host` to a reachable interface and supply
`auth_token`, following the official proxy's bind policy.

Closing an episode refuses new sampling calls and drains accepted calls.
`episode.turns` then contains immutable token/logprob records in submission
order. `turns_to_trajectory(episode.turns, stop_reason)` supplies cookbook
transitions; rewards and episode outcomes belong to the trial runner.
Cookbook decides which exact token-prefix continuations can share a Datum.

The recorder saves results synchronously before the official proxy parses
them. It does not use the capture exporter's lossy queue as training storage.
Sampling cancellation, missing logprobs or lost responses invalidate the
trajectory. Backend request rejections can be recovered by the agent without
inventing a sampled turn. No additional sampling retries, limits or parameter
overrides are introduced. Records are in memory: a worker crash discards the
unsubmitted rollout group rather than resuming a partial episode.

`tests/e2e/lora/test_tinker_client.py` installs the package in an isolated
environment and compares recorded requests/responses with the unwrapped
official proxy. The GPU gateway acceptance additionally trains from the
recorded tokens and logprobs.
