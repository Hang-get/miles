# Clef-style decision model training

This example trains a [Clef](https://huggingface.co/Cloudflare/clef)-style decision model:
a Qwen backbone plus Cloudflare's joint schema head. The head reads a structured
input (a state, plus one or more named questions with their allowed options) and
returns one probability distribution per question in a single forward pass. There
is no text generation and no reasoning trace.

Training is supervised. The loss is the Brier score between the head's option
probabilities and a target distribution:

$$L = \sum_i (p_i - t_i)^2$$

A target can be one-hot (a question with one correct answer) or soft (a random
outcome with a known distribution, such as a biased coin). Because Brier is a
proper scoring rule, it rewards calibrated probabilities rather than confident
guesses.

The head and record encoder in `joint_schema_model.py` are taken unchanged from
[Cloudflare/clef](https://huggingface.co/Cloudflare/clef) (revision
`2f3de3dd85f379784083b0814d997ab627200f0c`, Apache-2.0; see `LICENSE-Cloudflare`).

## Files

| File | Purpose |
|---|---|
| `train.py` | `torchrun` entrypoint: FSDP2, head warmup then joint training, validation, checkpoints |
| `joint_schema_model.py` | Cloudflare's head, record encoder, and release loader |
| `model.py` | Builds the Qwen backbone plus head and shards it with FSDP2 |
| `data.py` | Loads, validates, augments, and encodes labeled records |
| `objective.py` | Brier loss and validation metrics |
| `checkpoint.py`, `state.py`, `resume.py` | Resumable checkpoints and a serving export |
| `telemetry.py` | W&B, Prometheus, and Miles dashboard metrics |

## Data format

`--data-dir` must contain `train.jsonl` and `validation.jsonl`. Each line is one record:

```json
{
  "record": {
    "id": "coin-0001",
    "state": "A coin lands heads with probability 0.7. It will be flipped once.",
    "questions": {
      "answer": {
        "type": "choice",
        "instructions": "Take a guess at the outcome of the flip.",
        "criteria": {"A": "Heads", "B": "Tails"}
      }
    }
  },
  "targets": {"answer": {"A": 0.7, "B": 0.3}},
  "source": "synthetic_coin"
}
```

- `record` is the model input, in Clef's schema format. Question types are
  `choice` (named options), `noul` (true/false), and `score` (ordered levels).
- `targets` holds one distribution per question, keyed by option. Each must be
  finite, nonnegative, and sum to one. Targets are never part of the input.
- `source` groups metrics by data source.

Record IDs must be unique, and train/validation IDs must not overlap. Records
longer than `--max-length` tokens are rejected rather than truncated.

During training, options of a single `answer` question are shuffled together with
their targets at every visit. With probability `--multi-field-fraction` (default
0.25), a second true/false question asks whether a random option is the answer;
its target is that option's probability. Loss is averaged over questions within a
record, then over records.

## Launch

Run on every node, changing `--node-rank`. Checkpoint storage must be shared by
all ranks.

```bash
torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --master-addr=<rank-0-address> --master-port=29550 \
  -m examples.experimental.clef.train \
  --model-dir /models/Qwen3.8-27B \
  --data-dir /data/clef \
  --output-dir /outputs/clef-run \
  --checkpoint-dir /shared/checkpoints/clef-run \
  --run-name clef-example \
  --global-batch-size 64 --micro-batch-size 1 \
  --head-warmup-steps 128 --epochs 2 \
  --backbone-lr 3e-7 --head-lr 1e-5 \
  --save-interval 128 --eval-interval 32 \
  --wandb-project <project> --wandb-entity <entity>
```

- **Fresh head:** the head is randomly initialized and trained alone for
  `--head-warmup-steps` updates with the backbone frozen. Then the full text
  backbone and head train together for `--epochs` epochs. The vision encoder is
  kept but frozen. No LoRA is used.
- **Continue from a trained Clef export:** pass `--pretrained-head
  --head-warmup-steps 0` and point `--model-dir` at the export.
- **Resume:** pass `--resume <checkpoint-dir>` with the original configuration.
  Use `--max-steps` to extend a finished run.
- **Precision:** FP32 master weights and Adam states, BF16 compute, FP32
  gradient reduction. Gradients are clipped at `--max-grad-norm` (default 1).

Optional `--forecastbench-path` evaluates a second, held-out file in the same
record format (for example, resolved yes/no forecasting questions) at every
validation step. It is never used for training. Metrics are logged under
`forecastbench/`, including `binary_brier`, which is half the two-option Brier
score.

## Outputs

- `checkpoints/step_<n>/native`: sharded model and optimizer state, for resuming.
- `checkpoints/step_<n>/hf`: a serving export with backbone safetensors,
  `joint_head.safetensors`, `joint_head_config.json`, and the processor. It has
  the same layout as the released Clef checkpoints.
- A checkpoint is complete only once it contains `COMPLETE.json`.
- `validation/step_<n>.jsonl`: every predicted distribution.
- `traces/rank_<r>.jsonl`: input token IDs, targets, and predictions for each
  training record.

Logged metrics include Brier (record-weighted and field-weighted), single-answer
accuracy, expected calibration error, confidently wrong answers, and the share
of near-one-hot distributions, overall and per source. To view the Miles dashboard:

```bash
python -m miles.dashboard.serve --dump-details <output-dir> --follow --port 7788
```

## Serving

Recent SGLang versions serve Clef checkpoints natively: they detect
`joint_head_config.json`, load the head, and expose `/v1/systemone`. Launch the
`hf/` export like the released model; see the
[SGLang Clef cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/Cloudflare/clef.mdx).

`joint_schema_model.py` also provides `load_release_model` and `systemone` for
direct inference with Transformers.
