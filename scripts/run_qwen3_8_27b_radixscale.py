"""radixscale training backend under miles: Qwen3.8-27B / Qwen3.6-35B-A3B GRPO on dapo-math-17k.

miles owns rollout, reward, advantages and weight sync; radixscale (FSDP x EP, packed
thd) owns the training step. One tp4 sglang engine per node. --train-backend megatron
runs the identical recipe on Megatron (tp4 inside a node, HF weights loaded directly)
for the A/B.

Usage:
    python3 scripts/run_qwen3_8_27b_radixscale.py --num-nodes 4
    python3 scripts/run_qwen3_8_27b_radixscale.py --num-nodes 4 --train-backend megatron
"""

from dataclasses import dataclass
from typing import Literal

import typer

import miles.utils.external_utils.command_utils.legacy as U

MEGATRON_MODEL_TYPE = {"Qwen3.8-27B": "qwen3.8-27B", "Qwen3.6-35B-A3B": "qwen3.6-35B-A3B"}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    train_backend: Literal["radixscale", "megatron"] = "radixscale"
    model_name: str = "Qwen3.8-27B"
    ep_size: int = 1
    # parallel/perf knobs, tuned per backend for the A/B
    tp_size: int = 4
    recompute: bool = True
    compile: bool = False
    max_tokens_per_gpu: int = 16384
    fsdp_size: int = 0
    # trainer allocator; conflicts with the cuda_ipc weight-sync protocol, so train-only runs
    expandable_segments: bool = False
    # raw PYTORCH_CUDA_ALLOC_CONF for the trainer (allocator fragmentation experiments)
    alloc_conf: str = ""
    num_gpus_per_node: int = 4
    num_rollout: int = 20
    rollout_batch_size: int = 16
    n_samples_per_prompt: int = 8
    # fully async: half the nodes train, half serve rollouts continuously (train_async.py)
    fully_async: bool = False
    # "rollout": generate and save rollouts only; "train": replay saved rollouts, no engines
    debug_mode: Literal["none", "rollout", "train"] = "none"
    debug_data: str = "/root/shared_data/debug_rollout/qwen3.8-27b_{rollout_id}.pt"
    rollout_max_response_len: int = 8192
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    wandb_project: str = ""
    wandb_team: str = ""
    wandb_group: str = ""
    extra_args: str = ""


def execute(args: ScriptArgs):
    model_path = f"{args.model_dir}/{args.model_name}"
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
        f"--wandb-group {args.wandb_group or args.model_name.lower() + '-' + args.train_backend} "
        if args.wandb_project
        else ""
    )

    sglang_args = (
        f"--rollout-num-gpus-per-engine {args.num_gpus_per_node} "
        f"--sglang-mem-fraction-static {0.7 if args.fully_async else 0.35} "
        "--sglang-decode-log-interval 200 "
    )
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
            f"{'--radixscale-compile ' if args.compile else ''}"
            f"--radixscale-fsdp-size {args.fsdp_size} "
            "--log-probs-chunk-size 4096 "
            f"--expert-model-parallel-size {args.ep_size} "
            "--update-weight-buffer-size 2147483648 "
        )
    else:
        train_backend_args = (
            "--train-backend megatron "
            f"--ref-load {model_path} "
            f"--tensor-model-parallel-size {args.tp_size} "
            f"{'--sequence-parallel ' if args.tp_size > 1 else ''}"
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"{'--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 ' if args.recompute else ''}"
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
        "rollout": f"--debug-rollout-only --save-debug-rollout-data {args.debug_data} ",
        "train": f"--debug-train-only --load-debug-rollout-data {args.debug_data} ",
    }[args.debug_mode]

    alloc_conf = args.alloc_conf or ("expandable_segments:True" if args.expandable_segments else "")
    if alloc_conf:
        train_backend_args += f"""--train-env-vars '{{"PYTORCH_CUDA_ALLOC_CONF":"{alloc_conf}"}}' """

    if args.fully_async:
        actor_nodes = args.num_nodes // 2
        misc_args = (
            f"--actor-num-nodes {actor_nodes} "
            f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
            f"--num-gpus-per-node {args.num_gpus_per_node} "
            f"--rollout-num-gpus {(args.num_nodes - actor_nodes) * args.num_gpus_per_node} "
        )
    else:
        misc_args = (
            f"--actor-num-nodes {args.num_nodes} "
            f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
            f"--num-gpus-per-node {args.num_gpus_per_node} "
            "--colocate "
        )

    U.execute_train(
        train_args=(
            f"{rollout_args} {grpo_args} {optimizer_args} {wandb_args} "
            f"{sglang_args} {train_backend_args} {debug_args} {misc_args} {args.extra_args} "
        ),
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=MEGATRON_MODEL_TYPE[args.model_name] if args.train_backend == "megatron" else None,
        train_script="train_async.py" if args.fully_async else "train.py",
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
