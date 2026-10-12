"""radixscale training backend under miles: Qwen3.8-27B GRPO on dapo-math-17k, colocated.

miles owns rollout, reward, advantages and weight sync; radixscale (FSDP x EP, packed
thd) owns the training step. One tp4 sglang engine per node.

Usage:
    python3 scripts/run_qwen3_8_27b_radixscale.py --num-nodes 4
"""

from dataclasses import dataclass

import typer

import miles.utils.external_utils.command_utils.legacy as U

MODEL_NAME = "Qwen3.8-27B"


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_gpus_per_node: int = 4
    num_rollout: int = 20
    rollout_batch_size: int = 8
    n_samples_per_prompt: int = 4
    rollout_max_response_len: int = 8192
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    wandb_project: str = ""
    wandb_team: str = ""
    wandb_group: str = "qwen3.8-27b-radixscale"
    extra_args: str = ""


def execute(args: ScriptArgs):
    model_path = f"{args.model_dir}/{MODEL_NAME}"
    samples = args.rollout_batch_size * args.n_samples_per_prompt

    rollout_args = (
        f"--hf-checkpoint {model_path} "
        f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--balance-data "
        "--rm-type deepscaler "
        f"--num-rollout {args.num_rollout} "
        f"--rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        "--rollout-temperature 1 "
        f"--global-batch-size {samples} "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-team {args.wandb_team} "
        f"--wandb-group {args.wandb_group} "
        if args.wandb_project
        else ""
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        "--sglang-mem-fraction-static 0.35 "
        "--sglang-decode-log-interval 200 "
    )

    train_backend_args = (
        "--train-backend radixscale "
        "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
        "--gradient-checkpointing "
        "--update-weight-buffer-size 2147483648 "
    )

    misc_args = (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--colocate "
    )

    U.execute_train(
        train_args=(
            f"{rollout_args} {grpo_args} {optimizer_args} {wandb_args} "
            f"{sglang_args} {train_backend_args} {misc_args} {args.extra_args} "
        ),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=None,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
