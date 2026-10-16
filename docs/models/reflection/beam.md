---
title: "Beam × Miles"
description: "Full-parameter BF16 GRPO for Beam 501B on DAPO-Math-17K, with SGLang rollouts, Megatron training, AIME25 eval, and checkpoint resume."
---

Train Reflection's **Beam 501B** mixture-of-experts model with Miles using
[DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)](https://arxiv.org/abs/2503.14476).
Megatron trains the full model in BF16, SGLang generates responses, and Miles connects
the two: sample responses, score them, update the model, and send the new weights
back to the rollout engines.

This guide runs GRPO on DAPO-Math-17K with AIME25 evaluation on **96 H200
GPUs**.

## Before you start

You need twelve nodes with eight GPUs each, running the same image in an
existing Ray cluster:

- **64 GPUs for training:** eight nodes running Megatron.
- **32 GPUs for rollout:** two SGLang engines, each spread across sixteen GPUs.
- **Node-local NVMe:** at least 2 TB free on each trainer node, mounted at
  `/scratch`, for optimizer state.
- **Shared storage:** the model checkpoint and tokenizer, datasets, and a
  persistent output directory, available at the same paths on every node.

`prepare` downloads `reflection/Beam-1.0-501B-A23B` into `--model-dir`
(default `/root/models/Beam`). Use `--model-id` to select another compatible
Beam repository. The download requires access to that Hugging Face repository.
Training requires a local checkpoint directory, not a Hugging Face repository
ID; pass the same `--model-dir` to `prepare` and `execute`. The launcher loads the
checkpoint through Megatron Bridge; a separate offline checkpoint conversion
is not required.

## Get started

### 1. Build the image

From the repository root:

```bash
docker build -f docker/Dockerfile.beam -t miles-beam .
```

The Dockerfile starts from the Miles v0.1.2 image and replaces its SGLang with
Beam SGLang, pinned to the public `beam-miles` branch tip `9ac1c65b`: later
commits require torch 2.14.1, while the Miles v0.1.2 image is built on
torch 2.13 (`--build-arg BEAM_SGLANG_REF=<commit>` overrides the pin).
Beam uses the Miles model and Bridge plugins. Use this image on all twelve nodes.
The launcher submits to your Ray cluster; it does not allocate nodes or start
the cluster for you.

### 2. Prepare the model and data

Download the Beam checkpoint, DAPO-Math-17K for training, and AIME25 for
evaluation onto shared storage:

```bash
python scripts/run_beam.py prepare \
  --model-id reflection/Beam-1.0-501B-A23B \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets
```

Each row holds a chat-message `prompt` and an integer-answer `label`. Miles
renders the prompts with the checkpoint's chat template. `prepare` also writes
`aime-2025/aime-2025-answer-format.jsonl`, the AIME25 problems wrapped in the
answer-format instruction of the training prompts; evaluation reads this file.

### 3. Start training

Inside the image, from `/root/miles`, set the Ray head address and your mounted
paths. Replace `10.0.0.1` with the head node's address:

```bash
export MASTER_ADDR=10.0.0.1
export RAY_ADDRESS=http://${MASTER_ADDR}:8265

python scripts/run_beam.py execute \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets \
  --output-dir /root/shared_data/beam-run \
  --num-rollout 100
```

Miles publishes the initial weights, then repeatedly generates and scores
responses, updates the model, and publishes the new weights. It evaluates on
AIME25 every five rollouts and saves a checkpoint every twenty. Optimizer
scratch defaults to `/scratch/beam-optimizer`; use `--optimizer-disk-dir` to
change it. Keep the output directory on persistent shared storage so a later
job can resume.

### 4. Resume training

With the same model, data, output path, and training topology:

```bash
python scripts/run_beam.py execute \
  --model-dir /root/models/Beam \
  --data-dir /data/datasets \
  --output-dir /root/shared_data/beam-run \
  --resume \
  --num-rollout 200
```

Run this from an image with `MASTER_ADDR` and `RAY_ADDRESS` set as above. Miles
restores the latest checkpoint and the dataset's sampling position.
**`--num-rollout` is the total count**, including completed rollouts.

## The recipe

The defaults in `scripts/run_beam.py`:

- **Batch:** 32 prompts per rollout, eight responses each, one update of 256
  samples. A smaller `--global-batch-size` splits the update while still using
  every response once.
- **Lengths:** up to 1,024 prompt tokens and 4,096 response tokens, truncated at
  the limit.
- **Training:** tensor parallelism 8, expert parallelism 8, expert tensor
  parallelism 8, sequence parallelism, and full recompute. Pipeline and context
  parallelism are 1.
- **Optimizer:** distributed Adam at a constant learning rate of `1e-6`, with
  BF16 gradients, FP32 master weights, and BF16 moments streamed through NVMe.
- **Objective:** group-relative advantages with truncated importance sampling,
  and no KL or entropy penalty.
- **Reward:** the boxed answer is checked against the label, with a soft penalty
  for responses near the length limit.
- **Asynchronous rollout:** the next rollout is generated while the current one
  trains, so responses are one update behind. `--no-async-training` runs them
  in turn.
- **Evaluation:** AIME25 with eight samples per problem, reported as accuracy.
  Its prompts carry the training prompts' instruction to end with a boxed
  answer, which the grader requires.
- **Routing replay:** disabled by default. Pass `--use-rollout-routing-replay`
  to enable rollout routing replay together with the Miles router.
- **Dumps:** rollout, evaluation, and training data go to
  `<output-dir>/dumps`.

The launcher uses the standard Miles training drivers and native NVMe optimizer.
No training dataset is bundled.

These defaults provide a minimal starting point for running the training loop.

## Colocated variant

`python scripts/run_beam.py execute-colocate` runs the same recipe on **eight
nodes (64 H200 GPUs)**, with the same arguments as `execute`. Every GPU trains
and also hosts one of four rollout engines, each spread across sixteen GPUs, and
the new weights reach the engines over CUDA IPC. While the engines generate, the
paused trainer is written to the node-local NVMe that holds the optimizer
state, so rollout and training take turns: training runs synchronously.

```bash
MASTER_ADDR=10.0.0.1 RAY_ADDRESS=http://10.0.0.1:8265 \
python scripts/run_beam.py execute-colocate --num-rollout 100 \
  --model-dir /root/models/Beam --data-dir /data/datasets \
  --output-dir /root/shared_data/beam-colocate-run
```
