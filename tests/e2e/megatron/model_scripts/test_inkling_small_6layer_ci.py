import os
import zlib

if os.getenv("MILES_HARDWARE_PLATFORM") == "rocm":
    from scripts.amd.run_inkling import _MODEL_REGISTRY, ScriptArgs, _train
else:
    from scripts.run_inkling import _MODEL_REGISTRY, ScriptArgs, _train

from tests.ci.ci_register import register_cuda_ci, register_rocm_ci


# DNM verification of radixark/miles#4114 (--skip-actor-forward-only with
# --custom-model-provider-path). Run A keeps the standalone actor forward and
# logs step-0 ppo_kl / pg_clipfrac; run A2 repeats it on run A's rollout as the
# noise floor; run B replays the same rollout with the forward skipped. A2 and B
# must reproduce run A's grad norm.
register_cuda_ci(
    est_time=1800,
    suite="stage-c-4-gpu-h200",
    labels=["megatron", "model-scripts"],
    hardware=["hopper", "blackwell"],
    num_gpus=4,
)
register_rocm_ci(
    est_time=1800,
    suite="stage-c-4-gpu-mi350",
    labels=["megatron", "model-scripts", "amd"],
)

_MODEL_ORG = "pb09204048"
_VERIFY_DIR = "/tmp/inkling_skip_fwd_verify"
_ROLLOUT_DATA = f"{_VERIFY_DIR}/rollout_{{rollout_id}}.pt"
_GRAD_NORM = f"{_VERIFY_DIR}/grad_norm.pt"
_REWARD_FUNC = "tests.e2e.megatron.model_scripts.test_inkling_small_6layer_ci.reward_func"
_TRAIN_ONLY = (
    f"--load-debug-rollout-data {_ROLLOUT_DATA} --debug-train-only "
    f"--ci-load-grad-norm {_GRAD_NORM} --no-enable-sample-ownership-checker "
)


async def reward_func(args, sample, **kwargs):
    # The 6-layer slice never solves dapo-math, so math rewards are all 0 and
    # every advantage vanishes; a response hash gives nonzero advantages.
    return float(zlib.crc32(sample.response.encode()) % 2)


def _args(extra_args: str = "") -> ScriptArgs:
    return ScriptArgs.from_env(
        model_name="Inkling-Small-6layer",
        train_mode="full",
        task="dapo_math",
        num_nodes=1,
        num_gpus_per_node=4,
        rollout_num_gpus_per_engine=4,
        num_rollout=1,
        rollout_batch_size=4,
        n_samples_per_prompt=4,
        global_batch_size=16,
        rollout_max_response_len=512,
        sglang_context_length=1024,
        extra_args=(
            "--ci-test "
            "--ci-disable-kl-checker "
            "--check-weight-update-skip-list visual. audio. "
            "--ci-disable-logprobs-checker "
            "--offload-train-target cpu "
            f"{extra_args}"
        ),
    )


def prepare(args: ScriptArgs):
    U = args.create_backend()
    U.exec_command_cpu(f"mkdir -p {args.model_dir} {args.data_dir}")
    U.exec_command_cpu(f"hf download {_MODEL_ORG}/{args.model_name} --local-dir {args.hf_checkpoint}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=_MODEL_REGISTRY[args.model_name],
        num_gpus_per_node=args.num_gpus_per_node,
        dir_dst=args.model_dir,
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    prepare(_args())
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    os.makedirs(_VERIFY_DIR, exist_ok=True)
    print("[skip-fwd-verify] run A: standalone actor forward")
    execute(
        _args(
            f"--custom-rm-path {_REWARD_FUNC} "
            f"--save-debug-rollout-data {_ROLLOUT_DATA} --ci-save-grad-norm {_GRAD_NORM} "
        )
    )
    print("[skip-fwd-verify] run A2: standalone actor forward on run A's rollout")
    execute(_args(_TRAIN_ONLY))
    print("[skip-fwd-verify] run B: --skip-actor-forward-only on run A's rollout")
    execute(_args(f"{_TRAIN_ONLY}--skip-actor-forward-only "))
