import argparse
from dataclasses import dataclass

from pydantic import JsonValue

from miles.utils.test_utils.snapshot_comparison_models import SnapshotFieldMove


@dataclass(frozen=True)
class _Ownership:
    backend_fields: dict[str, frozenset[str]]
    shared_fields: frozenset[str]
    trainer_fields: frozenset[str]
    sglang_fields: frozenset[str]
    router_fields: frozenset[str]
    checkpoint_fields: frozenset[str]


def _complete_moves(
    *, snapshots: dict[str, JsonValue], ownership: _Ownership | None = None
) -> list[SnapshotFieldMove]:
    ownership = _load_ownership() if ownership is None else ownership
    moves: list[SnapshotFieldMove] = []
    for case_name, case in snapshots.items():
        assert isinstance(case, (dict, str)) or case is None
        if case_name.startswith("parsed_config/") and isinstance(case, dict) and "config" in case:
            assert isinstance(args := case["config"], dict)
            moves.extend(_scope_moves(args=args, scope=[case_name, "config"], ownership=ownership))
        elif case_name.startswith("runtime_config/"):
            assert isinstance(case, dict)
            for process_name, process in case.items():
                assert isinstance(process, dict) and isinstance(stages := process["stages"], dict)
                for stage_name, stage in stages.items():
                    assert isinstance(stage, dict) and isinstance(args := stage["args"], dict)
                    moves.extend(
                        _scope_moves(
                            args=args,
                            scope=[case_name, process_name, "stages", stage_name, "args"],
                            ownership=ownership,
                            trainer=bool(process["ranks"])
                            or process_name.rsplit("/", 1)[-1].startswith("trainer_controller_"),
                            checkpoint=stage_name.startswith("checkpoint_load-"),
                            runtime=True,
                        )
                    )
    return moves


def _scope_moves(
    *,
    args: dict[str, JsonValue],
    scope: list[str],
    ownership: _Ownership,
    trainer: bool = False,
    checkpoint: bool = False,
    runtime: bool = False,
) -> list[SnapshotFieldMove]:
    backend = args["train_backend"]
    assert isinstance(backend, str) and backend in ownership.backend_fields
    moves: list[SnapshotFieldMove] = []
    fields = ownership.backend_fields[backend]
    target = ("raw_megatron", "base_args") if backend == "megatron" else (f"raw_{backend}",)
    if trainer:
        target = ("backend",)
        fields |= ownership.trainer_fields
        if backend == "megatron":
            fields = fields | ownership.checkpoint_fields if checkpoint else fields - ownership.checkpoint_fields
    elif runtime and scope[1].rsplit("/", 1)[-1] in {"inference_controller", "multi_lora_controller"}:
        fields = frozenset()
    for name in sorted(fields & args.keys()):
        moves.append(
            SnapshotFieldMove(
                source=[*scope, name],
                target=[*scope, *target, name],
                op="op30-30/47/64;op48-130/131",
                reason="Backend parser declarations and TrainerBackendTraitConfig define ownership; values are unchanged",
                keep_source=not trainer and name in ownership.shared_fields,
            )
        )
    for prefix, names, target, op in (
        ("sglang_", ownership.sglang_fields, ("sglang", "base_args"), "op30-6"),
        ("router_", ownership.router_fields, ("router_args",), "op30-28"),
    ):
        for source in sorted(names):
            if source in args:
                moves.append(
                    SnapshotFieldMove(
                        source=[*scope, source],
                        target=[*scope, *target, source.removeprefix(prefix)],
                        op=op,
                        reason="The registered prefixed parser destination moves to the corresponding config dictionary",
                    )
                )
    return moves


def _load_ownership() -> _Ownership:
    from megatron.training.arguments import add_megatron_arguments

    from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser
    from miles.backends.megatron_utils.checkpoint_request import CHECKPOINT_LOAD_FIELDS
    from miles.backends.sglang_utils.sglang_config import _add_prefixed_primary_server_args
    from miles.backends.torchtitan_utils.arguments import TorchtitanArgs
    from miles.utils.args.configs.backend_fields import TrainerBackendTraitConfig
    from miles.utils.args.configs.router import RouterConfig
    from miles.utils.args.runtime import AllConfig

    megatron = argparse.ArgumentParser(add_help=False)
    add_megatron_arguments(megatron)
    parsers = {
        "megatron": megatron,
        "fsdp": build_dataclass_parser(FSDPArgs, "snapshot ownership"),
        "torchtitan": build_dataclass_parser(TorchtitanArgs, "snapshot ownership"),
    }
    derived = frozenset(
        {
            "rank",
            "world_size",
            "padded_vocab_size",
            "params_dtype",
            "data_parallel_size",
            "virtual_pipeline_model_parallel_size",
            "consumed_train_samples",
            "consumed_valid_samples",
            "skipped_train_samples",
            "variable_seq_lengths",
            "moe_router_use_torch_mm",
            "compress_ratios",
            "indexer_rope_interleave",
            "encoder_tensor_model_parallel_size",
            "encoder_pipeline_model_parallel_size",
            "_is_global_batch_size_explicitly_specified",
            "transformer_pipeline_model_parallel_size",
            "use_dist_ckpt",
            "use_layer_wise_distributed_optimizer",
            "model_type",
            "train_iters",
        }
    )
    non_megatron = frozenset(
        {
            "bf16",
            "calculate_per_token_loss",
            "ckpt_step",
            "clip_grad",
            "disable_grad_buffers_cpu_backup",
            "disable_param_buffers_cpu_backup",
            "finetune",
            "load",
            "mtp_detach_heads",
            "mtp_loss_scaling_factor",
            "mtp_num_layers",
            "no_load_optim",
            "no_load_rng",
            "no_save_optim",
            "rank",
            "world_size",
        }
    )
    sglang = argparse.ArgumentParser(add_help=False)
    _add_prefixed_primary_server_args(sglang)
    return _Ownership(
        backend_fields={
            name: frozenset(action.dest for action in parser._actions)
            | (derived if name == "megatron" else non_megatron)
            for name, parser in parsers.items()
        },
        shared_fields=frozenset(AllConfig.model_fields),
        trainer_fields=frozenset(TrainerBackendTraitConfig.model_fields),
        sglang_fields=frozenset(action.dest for action in sglang._actions),
        router_fields=frozenset(RouterConfig.arg_names()),
        checkpoint_fields=frozenset(CHECKPOINT_LOAD_FIELDS),
    )
