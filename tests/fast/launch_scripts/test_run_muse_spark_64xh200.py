import shlex
import sys

import pytest
import torch
from tests.fast.launch_scripts.model_args_harness import expand_model_args
from tests.fast.launch_scripts.py_harness import import_launch_script

import miles.utils.external_utils.command_utils.legacy as U


@pytest.mark.parametrize(
    "layout,optimizer", [("tp8_pp8_cp1", "adam"), ("tp4_pp8_cp2", "muon"), ("tp4_pp4_cp4_ep16", "muon")]
)
@pytest.mark.parametrize("counts", [(3, 8, 8), (100, 32, 16)])
@pytest.mark.parametrize("wandb_mode", [None, "online", "offline"])
def test_dapo_recipe_survives_megatron_validation(monkeypatch, tmp_path, wandb_mode, counts, layout, optimizer):
    from miles.backends.megatron_utils.arguments import parse_args, set_default_megatron_args, validate_args
    from miles.utils.arguments import get_miles_extra_args_provider, miles_validate_args

    launcher = import_launch_script(U.repo_base_dir / "scripts/run_muse_spark_64xh200.py")
    template = tmp_path / "chat.jinja"
    template.write_text("{{ messages }}")
    use_wandb = wandb_mode is not None
    recipe = launcher.ScriptArgs(
        training_layout=layout,
        optimizer=optimizer,
        reasoning_effort="xhigh" if optimizer == "muon" else None,
        rollout_max_response_len=4096 if optimizer == "muon" else 2048,
        use_rollout_routing_replay=layout == "tp4_pp4_cp4_ep16",
        max_tokens_per_gpu=2048 if layout == "tp4_pp4_cp4_ep16" else 4096,
        update_weight_buffer_size=4 * 1024**3 if layout == "tp4_pp4_cp4_ep16" else 512 * 1024**2,
        num_rollout=counts[0],
        rollout_batch_size=counts[1],
        n_samples_per_prompt=counts[2],
        chat_template_path=str(template),
        use_wandb=use_wandb,
        wandb_mode=wandb_mode or "online",
        wandb_project="muse-spark-m5" if use_wandb else None,
        wandb_team="radixarkai" if use_wandb else None,
    )
    tokens = shlex.split(launcher._training_args(recipe))
    assert "--use-dynamic-batch-size" in tokens and "--micro-batch-size" not in tokens
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr("megatron.training.arguments.get_device_arch_version", lambda: 9)
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setattr(sys, "argv", ["train.py", *expand_model_args("muse-spark-full"), *tokens])
    args = parse_args(extra_args_provider=get_miles_extra_args_provider())
    args.rank, args.world_size, args.compress_ratios = 0, 64, None
    args = set_default_megatron_args(args)
    miles_validate_args(args)
    validate_args(args)
    assert args.hf_checkpoint == "/personal/tmp-new-full"
    assert args.ref_load == "/personal/tmp-new-full_torch_dist"
    assert args.prompt_data == "/personal/dapo-math-17k/dapo-math-17k.jsonl"
    assert (args.rollout_batch_size, args.n_samples_per_prompt, args.global_batch_size) == (
        counts[1],
        counts[2],
        counts[1] * counts[2],
    )
    assert args.rollout_max_response_len == (4096 if optimizer == "muon" else 2048)
    if optimizer == "muon":
        assert args.apply_chat_template_kwargs["reasoning_effort"] == "xhigh"
    assert args.use_dynamic_batch_size and args.max_tokens_per_gpu == recipe.max_tokens_per_gpu
    assert args.update_weight_buffer_size == recipe.update_weight_buffer_size
    assert args.use_rollout_routing_replay == (layout == "tp4_pp4_cp4_ep16")
    assert args.use_routing_replay == args.use_rollout_routing_replay
    assert args.num_rollout == counts[0] and args.num_steps_per_rollout == 1
    assert args.dynamic_sampling_filter_path is None
    assert args.use_wandb == use_wandb
    if use_wandb:
        assert args.wandb_mode == wandb_mode and args.wandb_project == "muse-spark-m5"
        assert args.wandb_team == "radixarkai" and args.wandb_key is None
    tp, pp, cp, ep, edge_layers = {
        "tp8_pp8_cp1": (8, 8, 1, 8, 7),
        "tp4_pp8_cp2": (4, 8, 2, 8, 7),
        "tp4_pp4_cp4_ep16": (4, 4, 4, 16, 15),
    }[layout]
    assert (args.tensor_model_parallel_size, args.pipeline_model_parallel_size, args.expert_model_parallel_size) == (
        tp,
        pp,
        ep,
    )
    assert args.context_parallel_size == cp
    if cp > 1:
        from megatron.training.arguments import core_transformer_config_from_args

        from miles_plugins.models.muse_spark.provider import _validate_fused_context_parallel

        config = core_transformer_config_from_args(args)
        assert config.cp_comm_type == "a2a"
        _validate_fused_context_parallel(config)
    assert args.data_parallel_size == 1 and args.expert_tensor_parallel_size == 1
    assert args.num_layers == 62 and args.moe_layer_freq == [1] * 62
    assert args.decoder_first_pipeline_num_layers == args.decoder_last_pipeline_num_layers == edge_layers
    assert args.dist_ckpt_strictness == "raise_unexpected"
    if optimizer == "adam":
        assert args.optimizer == "adam" and args.stream_optimizer_state_to_disk
        assert args.stream_optimizer_state_moment_dtype == "fp32" and args.offload_train_target == "disk"
    else:
        assert args.optimizer == "muon" and args.use_layer_wise_distributed_optimizer
        assert not args.stream_optimizer_state_to_disk
        assert args.offload_train_target == "cpu" and args.chunked_optimizer_state_offload
        assert args.chunked_muon_master_weights and args.optimizer_state_offload_fraction == 1
    assert args.loss_type == "policy_loss" and args.advantage_estimator == "grpo"
    assert args.calculate_per_token_loss and not args.skip_actor_forward_only
    assert args.save is None and not args.debug_train_only and not args.debug_rollout_only
    assert (args.rollout_num_gpus_per_engine, args.sglang_ep_size, args.sglang_pp_size) == (16, 16, 1)
    assert not args.sglang_disable_cuda_graph and args.sglang_mem_fraction_static == 0.85
    assert args.sglang_max_running_requests == 64
    assert args.sglang_cuda_graph_bs_decode == [1, 2, 4, 8, 16, 32, 64]
    assert args.save_debug_rollout_data == recipe.run_dir + "/rollout_data/{rollout_id}.pt"


def test_new_entrypoint_uses_dapo_recipe(monkeypatch):
    launcher = import_launch_script(U.repo_base_dir / "scripts/run_muse_spark_64xh200.py")
    calls = []
    monkeypatch.setattr(U, "exec_command_cpu", lambda command: None)
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    launcher.execute(num_rollout=2)
    (call,) = calls
    tokens = shlex.split(call["train_args"])
    assert tokens[tokens.index("--num-rollout") + 1] == "2"
    assert tokens[tokens.index("--n-samples-per-prompt") + 1] == "8"
    assert call["train_script"] == "train.py" and len(call["ray_group"].node_addrs) == 8


@pytest.mark.parametrize("buffer_size", [4 * 1024**3, 8 * 1024**3])
def test_weight_buffer_override_reaches_train_entrypoint(monkeypatch, buffer_size):
    launcher = import_launch_script(U.repo_base_dir / "scripts/run_muse_spark_64xh200.py")
    calls = []
    monkeypatch.setattr(U, "exec_command_cpu", lambda command: None)
    monkeypatch.setattr(U, "execute_train", lambda **kwargs: calls.append(kwargs))
    launcher.execute(update_weight_buffer_size=buffer_size)
    tokens = shlex.split(calls[0]["train_args"])
    assert tokens.count("--update-weight-buffer-size") == 1
    assert tokens[tokens.index("--update-weight-buffer-size") + 1] == str(buffer_size)
