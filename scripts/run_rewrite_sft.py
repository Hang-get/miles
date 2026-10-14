"""Validate and submit a frozen two-epoch rewritten-trace SFT configuration."""

import io
import json
import os
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from megatron.training.arguments import core_transformer_config_from_args
from tap import Tap

from miles.utils.arguments import parse_args
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.external_utils.ray_job import run_ray_job


class Args(Tap):
    config: str
    preflight: bool = False


def main(cli: Args) -> None:
    path = Path(cli.config)
    config = json.loads(path.read_text())
    os.chdir(config["miles_path"])
    argv = shlex.split(load_model_args("qwen3.6-35B-A3B")) + config["argv"]
    if cli.preflight:
        with patch.object(sys, "argv", ["train_async.py", *argv]), redirect_stdout(io.StringIO()):
            args = parse_args()
            model = core_transformer_config_from_args(args)
        assert args.num_epoch == 2 and args.num_rollout == 1936
        assert args.save_interval == 968 and args.global_batch_size == args.rollout_batch_size == 4
        assert args.lr == args.min_lr == 1e-5 and args.lr_decay_style == "constant"
        assert args.mtp_num_layers == model.mtp_num_layers == 0 and not args.enable_mtp_training
        assert args.accumulate_allreduce_grads_in_fp32 and args.use_dynamic_global_batch_size
        assert args.seq_length == 262144 and args.debug_train_only
        audit = json.loads((path.parent / "data/mask-audit.json").read_text())
        assert audit["sessions"] == 3870 and audit["max_tokens"] <= 262144
        for repo, expected in config["source_commits"].items():
            actual = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True).strip()
            dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True).strip()
            assert actual == expected and not dirty, (repo, actual, dirty)
        result = {name: getattr(args, name) for name in (
            "num_epoch", "num_rollout", "save_interval", "lr", "min_lr", "optimizer",
            "global_batch_size", "seq_length", "mtp_num_layers", "accumulate_allreduce_grads_in_fp32",
            "tensor_model_parallel_size", "context_parallel_size", "expert_model_parallel_size")}
        (path.parent / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
        (path.parent / "resolved-argv.json").write_text(json.dumps(argv, indent=2) + "\n")
        print(json.dumps(result))
        return
    if not (path.parent / "preflight.json").exists():
        raise ValueError("Run preflight first")
    run_ray_job(entrypoint="python3 train_async.py " + shlex.join(argv),
                runtime_env={"env_vars": config["runtime_env"]})


if __name__ == "__main__":
    main(Args().parse_args())
