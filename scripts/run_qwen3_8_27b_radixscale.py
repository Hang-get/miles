"""GRPO on dapo-math-17k with the radixscale training backend, or the same recipe on Megatron.

Trains Qwen3.8-27B (dense) or Qwen3.6-35B-A3B (MoE) with one tp4 sglang engine per node,
colocated by default, fully async with --fully-async. Expects the HF checkpoint under
--model-dir and dapo-math-17k under --data-dir; radixscale must be importable on every node.

Args:
    --train-backend: radixscale or megatron.
    --model-name: Qwen3.8-27B or Qwen3.6-35B-A3B.
    --fsdp-size / --ep-size / --expert-rep: radixscale dense FSDP group, expert parallel and
        expert replica sizes (0 = one NVLink domain / no replicas).
    --tp-size: Megatron tensor parallel size.
    --recompute / --recompute-skip-layers: activation recompute, and the number of layers that
        keep their activations instead (radixscale).
    --debug-mode: rollout saves rollouts to --debug-data; train replays them without engines.

Examples:
    python3 scripts/run_qwen3_8_27b_radixscale.py --num-nodes 4 --model-name Qwen3.6-35B-A3B \
        --fsdp-size 16 --ep-size 4 --max-tokens-per-gpu 65536 --recompute-skip-layers 16
    python3 scripts/run_qwen3_8_27b_radixscale.py --num-nodes 4 --train-backend megatron --tp-size 2 --no-recompute
"""

from dataclasses import dataclass
from typing import Literal

import typer

from miles.utils.external_utils import command_utils

_MEGATRON_MODEL_TYPE = {"Qwen3.8-27B": "qwen3.8-27B", "Qwen3.6-35B-A3B": "qwen3.6-35B-A3B"}


@dataclass
class ScriptArgs(command_utils.ExecuteTrainConfig):
    train_backend: Literal["radixscale", "megatron"] = "radixscale"
    model_name: Literal["Qwen3.8-27B", "Qwen3.6-35B-A3B"] = "Qwen3.8-27B"
    fsdp_size: int = 0
    ep_size: int = 1
    expert_rep: int = 0
    tp_size: int = 4
    recompute: bool = True
    recompute_skip_layers: int = 0
    max_tokens_per_gpu: int = 16384
    num_gpus_per_node: int = 4
    num_rollout: int = 20
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    rollout_max_response_len: int = 8192
    fully_async: bool = False
    debug_mode: Literal["none", "rollout", "train"] = "none"
    debug_data: str = ""
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    wandb_project: str = ""
    wandb_team: str = ""
    wandb_group: str = ""
    extra_args: str = ""

    @property
    def debug_data_path(self) -> str:
        return self.debug_data or f"{self.output_dir}/debug_rollout/{self.model_name.lower()}_{{rollout_id}}.pt"


def execute(args: ScriptArgs) -> None:
    U = args.create_backend()
    model_path = f"{args.model_dir}/{args.model_name}"

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
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
    )
    grpo_args = "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    wandb_args = (
        f"--use-wandb --wandb-project {args.wandb_project} --wandb-team {args.wandb_team} --wandb-group {args.wandb_group or args.model_name.lower() + '-' + args.train_backend} "
        if args.wandb_project
        else ""
    )
    sglang_args = f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} --sglang-mem-fraction-static {0.7 if args.fully_async else 0.35} --sglang-decode-log-interval 200 "
    if args.fully_async:
        rollout_args += "--fully-async --pause-generation-mode in_place "
        grpo_args += "--use-tis "

    if args.train_backend == "radixscale":
        train_backend_args = (
            "--train-backend radixscale "
            "--micro-batch-size 1 "
            "--use-dynamic-batch-size "
            f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
            f"{'--gradient-checkpointing ' if args.recompute else ''}"
            f"--radixscale-recompute-skip-layers {args.recompute_skip_layers} "
            f"--radixscale-fsdp-size {args.fsdp_size} "
            f"--expert-model-parallel-size {args.ep_size} "
            f"--radixscale-expert-rep {args.expert_rep} "
            "--log-probs-chunk-size 4096 "
            "--update-weight-buffer-size 2147483648 "
        )
    else:
        recompute_args = "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        train_backend_args = (
            "--train-backend megatron "
            f"--ref-load {model_path} "
            f"--tensor-model-parallel-size {args.tp_size} "
            f"{'--sequence-parallel ' if args.tp_size > 1 else ''}"
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"{recompute_args if args.recompute else ''}"
            "--micro-batch-size 1 "
            "--use-dynamic-batch-size "
            f"--max-tokens-per-gpu {args.max_tokens_per_gpu} "
            "--use-distributed-optimizer "
            f"--expert-model-parallel-size {args.ep_size} "
            "--expert-tensor-parallel-size 1 "
            "--log-probs-chunk-size 4096 "
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend auto "
            "--update-weight-buffer-size 2147483648 "
        )

    debug_args = {
        "none": "",
        "rollout": f"--debug-rollout-only --save-debug-rollout-data {args.debug_data_path} ",
        "train": f"--debug-train-only --load-debug-rollout-data {args.debug_data_path} ",
    }[args.debug_mode]

    if args.fully_async:
        actor_nodes = args.num_nodes // 2
        misc_args = f"--actor-num-nodes {actor_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} --num-gpus-per-node {args.num_gpus_per_node} --rollout-num-gpus {(args.num_nodes - actor_nodes) * args.num_gpus_per_node} "
    else:
        misc_args = f"--actor-num-nodes {args.num_nodes} --actor-num-gpus-per-node {args.num_gpus_per_node} --num-gpus-per-node {args.num_gpus_per_node} --colocate "

    U.execute_train(
        train_args=(
            f"{rollout_args} {grpo_args} {optimizer_args} {wandb_args} {sglang_args} {train_backend_args} {debug_args} {misc_args} {args.extra_args} "
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=_MEGATRON_MODEL_TYPE[args.model_name] if args.train_backend == "megatron" else None,
        train_script="train_async.py" if args.fully_async else "train.py",
    )


@command_utils.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
