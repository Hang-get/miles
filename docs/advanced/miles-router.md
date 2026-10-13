---
title: Rollout Routing Replay (R3)
description: Capture expert routing during inference and replay it during training to stabilize RL.
---
Rollout Routing Replay (R3) records the expert routing decisions made during
inference and replays them during training, producing bit-identical expert
allocation between rollout and training.

## Why MoE RL is unstable without R3

For each token, an MoE router picks `top-k` experts. The choice depends on the
input through a soft router and a top-k operation. In production the router is a
learned `nn.Linear` with non-deterministic kernels and FP8 quantization, so tiny
numerical differences flip routes at the per-layer, per-token level.

An example without R3:

* Rollout selects experts `{2, 7}` for token 314.
* Training (with the same weights but slightly different precision and kernels)
  selects experts `{2, 8}` for token 314.
* The gradient is computed against the wrong expert. Multiplied by hundreds of
  layers, tens of thousands of tokens, and thousands of training steps, the
  policy diverges.

With R3, the trainer replays the rollout router's expert assignments verbatim,
so numerical noise no longer flips routes.

## How R3 wires up

**SGLang side.** Miles enables `enable_return_routed_experts` automatically when
`--use-rollout-routing-replay` is on. SGLang then includes `routed_experts`
in `meta_info` of each response, with shape `(seq_len - 1, num_layers, top_k)`.
By default it is base64 of the `int32` buffer. Add
`--sglang-routed-experts-encoding zstd` to send it sorted, narrowed to `uint8`
(`uint16` past 256 experts) and zstd-compressed: about 7x fewer bytes on the
wire and far less CPU in SGLang's detokenizer and tokenizer manager.
`meta_info["routed_experts_encoding"]` names the encoding, so Miles decodes
either.

**Miles side.** Enable R3 with:

```bash
--use-rollout-routing-replay
```

Rollout sends `return_routed_experts=true` in each request and stores the
results in `sample.rollout_routed_experts` (`miles/utils/types.py`) as `uint8`
(`uint16` past 256 experts; `miles/utils/routed_experts.py`). In multi-turn
rollouts, each turn after the first asks only for the rows the sample does not
hold yet (`routed_experts_start_len`), unless `--pause-generation-mode retract`
can recompute the prefix on a weight update. The
trainer pushes the arrays through `RoutingReplayManager`
(`miles/utils/replay_base.py`), and `replay_utils.py` plugs them into the
forward pass so recorded routes are used instead of recomputed ones.

## Memory cost

`(num_tokens - 1) × num_layers × top_k` bytes in rollout (one byte per
expert id up to 256 experts, two beyond). For a 32K-token sequence, 60 layers,
and `top_k = 8`, that is roughly 15 MB per sample of routing metadata. The
trainer widens each microbatch to `int32` before padding.

## When R3 is not required

* The model is dense.
* The `--advantage-estimator` is `reinforce_plus_plus` and `--use-tis` already
  masks the off-policy term.

For MoE training with `--advantage-estimator grpo`, current recipes turn R3
on (for example `scripts/run_glm47_flash.py`).

## References

* R3 paper: [arXiv 2510.11370](https://arxiv.org/pdf/2510.11370).
* Routing replay: [arXiv 2507.18071](https://arxiv.org/abs/2507.18071).
