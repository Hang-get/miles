"""Train full Muse Spark with GRPO on 64 H200 GPUs: DAPO-Math-17k, or Geometry3K images with --vlm.

Requires storage shared by all eight nodes under model_dir (and under output_dir to save weights), and
Muse-enabled Megatron/SGLang on every node. Join the Ray cluster before running with MILES_SCRIPT_EXTERNAL_RAY=1;
the first run downloads the HF checkpoint and converts it into an adjacent _torch_dist release on that cluster.
Adam streams optimizer state to local disk; Muon keeps state/master weights on CPU and computes
updates in GPU chunks. Each rollout performs one update.

Args:
    model_org / model_dir / model_name: HF repository <model_org>/<model_name>, downloaded to
        <model_dir>/<model_name>; its _torch_dist release is written next to it.
    data_dir / output_dir: Dataset root and run output root.
    training_layout: tp8_pp8_cp1, tp4_pp8_cp2, or tp4_pp4_cp4_ep16.
    optimizer / learning_rate: adam or muon, with learning rate 1e-6.
    num_rollout / rollout_batch_size / n_samples_per_prompt: Rollouts, prompts per rollout, samples per prompt.
    use_dynamic_sampling_filter: Keep only nonconstant-reward groups.
    rollout_max_response_len / rollout_max_context_len: Response and prompt-plus-response token limits.
    log_probs_chunk_size: Defaults 4096; -1 disables log-prob chunking.
    sglang_text_only: Serve only the language model from the omni checkpoint; defaults False.
    vlm: Train on geo3k images instead of DAPO-Math (frozen vision tower); defaults False.
    eval_interval / n_samples_per_eval_prompt: geo3k test-split eval cadence (vlm only) and samples per prompt.
    reasoning_effort: Optional chat-template override.
    save_interval: Save the weights without optimizer state every N rollouts and after the last one, under
        <run_dir>/checkpoints on storage shared by all eight nodes. After training, the last save is converted
        on CPU into an HF checkpoint in the released layout under <run_dir>/hf;
        tools/convert_muse_spark_torch_dist_to_hf.py converts any other save. Defaults None: nothing is saved.
    extra_args / extra_env_vars: Additional training flags and runtime environment.

Example:
    MILES_SCRIPT_EXTERNAL_RAY=1 python scripts/run_muse_spark_64xh200.py
"""

import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils.legacy as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    num_nodes: int = 8
    num_gpus_per_node: int = 8
    run_id: str = field(default_factory=U.create_run_id)
    model_org: str = "meta-models"
    model_name: str = "Muse-Spark-open-816B-A42B"
    megatron_model_type: str = "muse-spark-full"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    training_layout: Literal["tp8_pp8_cp1", "tp4_pp8_cp2", "tp4_pp4_cp4_ep16"] = "tp8_pp8_cp1"
    optimizer: Literal["adam", "muon"] = "adam"
    learning_rate: float = 1e-6
    num_rollout: int = 30
    rollout_batch_size: int = 32
    n_samples_per_prompt: int = 16
    use_dynamic_sampling_filter: bool = False
    rollout_max_response_len: int = 4096
    rollout_max_context_len: int = 8192
    use_dynamic_batch_size: bool = True
    max_tokens_per_gpu: int = 4096
    log_probs_chunk_size: int = 4096
    use_rollout_routing_replay: bool = True
    update_weight_buffer_size: int = 2 * 1024**3
    reasoning_effort: Literal["low", "medium", "high", "xhigh"] | None = "xhigh"
    chat_template_path: str | None = None
    offload_train_disk_dir: str | None = None
    sglang_mem_fraction_static: float = 0.8
    sglang_enable_dp_attention: bool = False
    # Per engine; SGLang splits it across attention-DP ranks when --sglang-dp-size > 1.
    sglang_max_running_requests: int = 64
    # Serve only the language model (skips the vision/audio towers of the omni checkpoint).
    sglang_text_only: bool = False
    # Image-grounded math on geo3k (scripts/muse_spark_vlm/prepare_geo3k.py under data_dir/geo3k): the
    # omni provider runs the frozen NaViT tower on the first pipeline stage, and the weight iterator re-sends the
    # towers from the HF checkpoint after every update.
    vlm: bool = False
    eval_interval: int | None = None
    n_samples_per_eval_prompt: int = 4
    save_interval: int | None = None
    extra_args: str = ""

    def __post_init__(self):
        if self.num_nodes != 8 or self.num_gpus_per_node != 8:
            raise ValueError("Full-model layouts require eight nodes with eight GPUs each")
        if self.optimizer == "muon" and self.offload_train_disk_dir is not None:
            raise ValueError("--offload-train-disk-dir requires --optimizer adam")
        if self.vlm and self.sglang_text_only:
            raise ValueError("--vlm needs the omni engine; drop --sglang-text-only")
        if self.eval_interval is not None and not self.vlm:
            raise ValueError("--eval-interval evaluates the geo3k test split and needs --vlm")
        if self.chat_template_path is None:
            # The checkpoint's template renders an image part as a bare <|patch|>; the repo template renders the
            # <image> placeholder the train processor expands (its text rendering is identical).
            self.chat_template_path = (
                str(
                    Path(__file__).resolve().parents[1]
                    / "miles/utils/chat_template_utils/templates/muse_spark_fixed.jinja"
                )
                if self.vlm
                else f"{self.model_dir}/{self.model_name}/chat_template.jinja"
            )
        if self.optimizer == "adam" and self.offload_train_disk_dir is None:
            self.offload_train_disk_dir = f"{self.run_dir}/offload"

    @property
    def run_dir(self) -> str:
        return f"{self.output_dir}/{self.run_id}"


# training_layout -> tensor, pipeline, context, expert parallel sizes and the decoder layers on the first/last stage.
_LAYOUTS = {
    "tp8_pp8_cp1": (8, 8, 1, 8, 7),
    "tp4_pp8_cp2": (4, 8, 2, 8, 7),
    "tp4_pp4_cp4_ep16": (4, 4, 4, 16, 15),
}


def _model_parallel_args(tp: int, pp: int, cp: int, ep: int, edge_layers: int) -> str:
    parallel_args = (
        f"--tensor-model-parallel-size {tp} --pipeline-model-parallel-size {pp} "
        f"--decoder-first-pipeline-num-layers {edge_layers} --decoder-last-pipeline-num-layers {edge_layers} "
        f"--context-parallel-size {cp} --expert-model-parallel-size {ep} "
        "--expert-tensor-parallel-size 1 "
    )
    if cp > 1:
        parallel_args += "--cp-comm-type a2a "
    return parallel_args


def prepare(args: ScriptArgs):
    hf_checkpoint = Path(args.model_dir) / args.model_name
    # hf download skips files that are already complete.
    U.exec_command_cpu(f"hf download {args.model_org}/{args.model_name} --local-dir {shlex.quote(str(hf_checkpoint))}")
    # Converts on all eight Ray nodes into <model_dir>/<model_name>_torch_dist; skipped once that release exists.
    # The release loads into any layout. Tensor/context parallelism would need CUDA_DEVICE_MAX_CONNECTIONS=1 on
    # every Ray node, so the conversion keeps only the pipeline and expert split of the training layout.
    _, pp, _, ep, edge_layers = _LAYOUTS[args.training_layout]
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.num_gpus_per_node,
        multinode=True,
        num_nodes=args.num_nodes,
        extra_args=_model_parallel_args(tp=1, pp=pp, cp=1, ep=ep, edge_layers=edge_layers),
        dir_dst=args.model_dir,
        hf_checkpoint=str(hf_checkpoint),
        megatron_path=args.megatron_path,
    )
    if not args.vlm:
        U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
        return
    script = Path(__file__).resolve().parent / "muse_spark_vlm/prepare_geo3k.py"
    for split in ("train", "test"):
        U.exec_command_cpu(
            f"python3 {shlex.quote(str(script))} --out {shlex.quote(str(Path(args.data_dir) / 'geo3k'))} "
            f"--split {split} --max-samples 1000000"
        )


def execute(args: ScriptArgs):
    ckpt_args = (
        f"--hf-checkpoint {shlex.quote(str(Path(args.model_dir) / args.model_name))} "
        f"--ref-load {shlex.quote(str(Path(args.model_dir) / (args.model_name + '_torch_dist')))} "
        f"--load {shlex.quote(args.run_dir + '/checkpoints')} "
        "--dist-ckpt-strictness raise_unexpected "
    )
    if args.save_interval is not None:
        ckpt_args += f"--save {shlex.quote(args.run_dir + '/checkpoints')} --save-interval {args.save_interval} "
    geo3k = Path(args.data_dir) / "geo3k"
    if args.vlm:
        rollout_args = (
            f"--prompt-data {shlex.quote(str(geo3k / 'train_rl.jsonl'))} "
            "--input-key messages "
            f"--multimodal-keys {shlex.quote(json.dumps({'image': 'images'}))} "
        )
    else:
        rollout_args = (
            f"--prompt-data {shlex.quote(str(Path(args.data_dir) / 'dapo-math-17k/dapo-math-17k.jsonl'))} "
            "--input-key prompt "
        )
    rollout_args += (
        "--label-key label "
        "--apply-chat-template "
        f"--chat-template-path {shlex.quote(args.chat_template_path)} "
        "--rollout-shuffle "
        "--rm-type math "
        "--rollout-temperature 1 "
        f"--num-rollout {args.num_rollout} --rollout-batch-size {args.rollout_batch_size} "
        f"--n-samples-per-prompt {args.n_samples_per_prompt} "
        f"--global-batch-size {args.rollout_batch_size * args.n_samples_per_prompt} "
        "--num-steps-per-rollout 1 "
        "--balance-data "
        f"--save-debug-rollout-data {shlex.quote(args.run_dir + '/rollout_data/{rollout_id}.pt')} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--rollout-max-context-len {args.rollout_max_context_len} "
        f"--rollout-max-prompt-len {args.rollout_max_context_len - args.rollout_max_response_len} "
    )
    if args.use_dynamic_sampling_filter:
        rollout_args += (
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.common_filters.apply_reward_nonzero_std_filter "
        )
    if args.reasoning_effort is not None:
        template_kwargs = json.dumps({"reasoning_effort": args.reasoning_effort})
        rollout_args += f"--apply-chat-template-kwargs {shlex.quote(template_kwargs)} "
    if args.eval_interval is not None:
        rollout_args += (
            f"--eval-prompt-data geo3k {shlex.quote(str(geo3k / 'test_rl.jsonl'))} "
            f"--eval-interval {args.eval_interval} --n-samples-per-eval-prompt {args.n_samples_per_eval_prompt} "
        )
    grpo_args = (
        "--loss-type policy_loss "
        "--advantage-estimator grpo "
        "--kl-coef 0 "
        "--kl-loss-coef 0 "
        "--entropy-coef 0 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--calculate-per-token-loss "
    )
    if args.use_rollout_routing_replay:
        grpo_args += "--use-rollout-routing-replay "
    perf_args = (
        f"{_model_parallel_args(*_LAYOUTS[args.training_layout])}"
        "--sequence-parallel "
        "--qkv-format thd "
        "--seq-length 8192 "
        "--attention-backend flash "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        f"--log-probs-chunk-size {args.log_probs_chunk_size} "
    )
    if args.use_dynamic_batch_size:
        perf_args += f"--use-dynamic-batch-size --max-tokens-per-gpu {args.max_tokens_per_gpu} "
    optimizer_args = (
        f"--optimizer {'adam' if args.optimizer == 'adam' else 'dist_muon'} "
        f"--lr {args.learning_rate} --min-lr {args.learning_rate} "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.95 "
        "--adam-eps 1e-8 "
        "--clip-grad 0 "
        "--use-distributed-optimizer "
        "--ckpt-format torch_dist "
        "--no-load-optim "
        "--no-save-optim "
    )
    if args.optimizer == "adam":
        optimizer_args += (
            "--stream-optimizer-state-to-disk "
            "--stream-optimizer-state-moment-dtype fp32 "
            f"--offload-train-disk-dir {shlex.quote(args.offload_train_disk_dir)} "
            "--offload-train-disk-chunk-mb 256 "
        )
    else:
        optimizer_args += (
            "--muon-momentum 0.9 "
            "--muon-num-ns-steps 5 "
            "--muon-scale-mode spectral "
            "--muon-tp-mode blockwise "
            "--chunked-optimizer-state-offload "
            "--optimizer-state-offload-fraction 1 "
            "--optimizer-state-offload-chunk-size-mb 256 "
            "--chunked-muon-master-weights "
        )
    sglang_args = (
        "--colocate "
        "--offload-train "
        "--offload-rollout "
        f"--offload-train-target {'disk' if args.optimizer == 'adam' else 'cpu'} "
        "--rollout-num-gpus-per-engine 16 "
        "--sglang-ep-size 16 "
        "--sglang-pp-size 1 "
        f"--sglang-dtype bfloat16 --sglang-mem-fraction-static {args.sglang_mem_fraction_static} "
        # SGLang would otherwise pick an FP8 KV cache, which widens the rollout/trainer log-prob gap.
        "--sglang-kv-cache-dtype bfloat16 "
        "--sglang-chunked-prefill-size 1024 "
        f"--sglang-context-length {args.rollout_max_context_len} "
        "--sglang-router-policy round_robin "
        "--sglang-cuda-graph-bs-decode 1 2 4 8 16 32 64 "
        "--sglang-cuda-graph-backend-prefill disabled "
        f"--sglang-max-running-requests {args.sglang_max_running_requests} "
    )
    if args.sglang_enable_dp_attention:
        sglang_args += "--sglang-enable-dp-attention --sglang-dp-size 2 "
    if args.sglang_text_only:
        # The omni config still carries vision/audio sub-configs, so the engine must also be told to serve
        # the language model only; otherwise it looks for a multimodal processor of the text architecture.
        text_only = json.dumps({"architectures": ["MuseSpark1xForCausalLM"]})
        sglang_args += f"--sglang-json-model-override-args {shlex.quote(text_only)} --sglang-language-model-only "
    if args.vlm:
        # Overrides the text provider of the model args, which precede the train args.
        misc_args = (
            "--custom-model-provider-path miles_plugins.models.muse_spark.provider.muse_spark_1x_omni_model_provider "
        )
    else:
        misc_args = ""
    misc_args += (
        f"--actor-num-nodes {args.num_nodes} "
        f"--actor-num-gpus-per-node {args.num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--update-weight-buffer-size {args.update_weight_buffer_size} "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{perf_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars={
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "1",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
        },
    )


def export_hf(args: ScriptArgs):
    # The last save is iteration num_rollout - 1; the conversion runs on CPU once training has finished.
    last_save = f"{args.run_dir}/checkpoints/iter_{args.num_rollout - 1:07d}"
    U.exec_command_cpu(
        f"PYTHONPATH={shlex.quote(f'{U.repo_base_dir}:{args.megatron_path}')}${{PYTHONPATH:+:$PYTHONPATH}} "
        f"python3 {shlex.quote(str(U.repo_base_dir / 'tools/convert_muse_spark_torch_dist_to_hf.py'))} "
        f"--input-dir {shlex.quote(last_save)} "
        f"--hf-checkpoint {shlex.quote(str(Path(args.model_dir) / args.model_name))} "
        f"--output-dir {shlex.quote(args.run_dir + '/hf')}"
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    prepare(args)
    execute(args)
    if args.save_interval is not None:
        export_hf(args)


if __name__ == "__main__":
    typer.run(main)
