"""Validate a frozen audited SFT configuration before submitting it to Ray."""

import io
import json
import os
import shlex
import subprocess
import sys
from contextlib import redirect_stdout
from math import ceil
from pathlib import Path
from unittest.mock import patch

from megatron.training.arguments import core_transformer_config_from_args
from tap import Tap

from miles.utils.arguments import parse_args
from miles.utils.external_utils.model_args_utils import load_model_args
from miles.utils.external_utils.ray_job import run_ray_job
from miles.utils.types import Sample
from miles.utils.weight_version import assert_samples_weight_version_sane


class Args(Tap):
    config: str
    preflight: bool = False


def main(cli: Args) -> None:
    path = Path(cli.config)
    config = json.loads(path.read_text())
    os.chdir(config["miles_path"])
    argv = shlex.split(load_model_args("qwen3.6-35B-A3B")) + config["argv"]
    if cli.preflight:
        with patch.object(sys, "argv", ["train.py", *argv]), redirect_stdout(io.StringIO()):
            args = parse_args()
            model = core_transformer_config_from_args(args)
        audit = json.loads((path.parent / "data/mask-audit.json").read_text())
        batches = ceil(audit["sessions"] / args.rollout_batch_size)
        assert args.num_epoch == config["epochs"]
        assert args.num_rollout == batches * args.num_epoch
        assert args.save_interval == batches
        assert args.global_batch_size == args.rollout_batch_size == 4
        assert args.lr == args.min_lr == 1e-5 and args.lr_decay_style == "constant"
        assert args.mtp_num_layers in (None, 0) and model.mtp_num_layers in (None, 0)
        assert not args.enable_mtp_training
        assert args.accumulate_allreduce_grads_in_fp32 and args.use_dynamic_global_batch_size
        assert args.seq_length == 262144 and args.debug_train_only
        assert args.pretrained_checkpoint == args.hf_checkpoint
        assert_samples_weight_version_sane(args, samples=[Sample(index=0, tokens=[1, 2], response_length=1)])
        assert audit["sessions"] == config["sessions"] and audit["max_tokens"] <= 262144
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
        (path.parent / "parsed-argument-names.json").write_text(json.dumps(sorted(vars(args)), indent=2) + "\n")
        print(json.dumps(result))
        return
    if not (path.parent / "preflight.json").exists():
        raise ValueError("Run preflight first")
    run_ray_job(entrypoint="python3 " + shlex.quote(str(Path(config["miles_path"]) / "train.py"))
                + " " + shlex.join(argv), runtime_env={"env_vars": config["runtime_env"]})


if __name__ == "__main__":
    main(Args().parse_args())
