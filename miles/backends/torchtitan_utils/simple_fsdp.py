"""``--titan-fsdp-impl simple_fsdp``: FSDP with its collectives inside each compiled TransformerBlock (TPU).

FSDP2 issues its all-gathers and reduce-scatters from module hooks, outside the compiled blocks, and TorchTPU
runs each eager collective as a program of its own, so they never overlap compute. SimpleFSDP (torchtitan
graph_trainer) makes every parameter a DTensor parametrization instead: the all-gather is a ``redistribute`` in
the block's forward and the reduce-scatter is its backward, so both are traced into the compiled block.
``torch.tpu.distributed.spmd_safe`` on the block's forward then keeps them in the same program as the block's
compute; without it torch_tpu's split compiler cuts the program at every collective.

Parameters are gathered and gradients reduce-scattered in every microbatch, which removes FSDP's per-call cost
but adds a partly hidden per-microbatch one: this pays off with few microbatches per call.
"""

import dataclasses
import functools
import logging
import types

import torch
from torch import nn

logger = logging.getLogger(__name__)


def wrap_parallelize_fn(base_fn):
    """The model spec's parallelize_fn with SimpleFSDP in place of FSDP2."""

    def parallelize(
        model, *, parallel_dims, training, parallelism, compile_config, ac_config, dump_folder, skip_dp=False
    ):
        for block in model.layers.values():
            block.forward = torch.tpu.distributed.spmd_safe(block.forward)
        # torchtitan would checkpoint-wrap each block and compile the wrapper; with the DTensor redistribute in
        # that graph, torch 2.13's AOT runtime fails saving a DeviceMesh for backward. Compile the block itself
        # and checkpoint it eagerly instead (same recompute, all-gathers included).
        model = base_fn(
            model,
            parallel_dims=parallel_dims,
            training=training,
            parallelism=parallelism,
            compile_config=dataclasses.replace(compile_config, enable=False),
            ac_config=None,
            dump_folder=dump_folder,
            skip_dp=True,
        )
        if not skip_dp:
            _apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)
        _install_block_forward(model, compile_config=compile_config, ac_config=ac_config)
        return model

    return parallelize


def _apply_simple_fsdp(model: nn.Module, *, parallel_dims, training) -> None:
    from torchtitan.config import TORCH_DTYPE_MAP
    from torchtitan.experiments.graph_trainer.simple_fsdp import (
        MixedPrecisionPolicy,
        data_parallel,
        disable_active_parametrization,
    )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled or parallel_dims.cp_enabled or parallel_dims.pp_enabled:
        raise ValueError("--titan-fsdp-impl simple_fsdp supports data parallelism only")
    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    data_parallel(
        model,
        parallel_dims.get_mesh(names),
        mode="hybrid_shard" if parallel_dims.dp_replicate_enabled else "fully_shard",
        mp_policy=MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        ),
    )
    # init must see the sharded Parameters: weight tying assigns lm_head's Parameter to the embedding, and a
    # parametrized read would return the gathered plain tensor
    init_states = model.init_states

    def init_states_unparametrized(self, **kwargs):
        with disable_active_parametrization():
            return init_states(**kwargs)

    model.init_states = types.MethodType(init_states_unparametrized, model)


def _install_block_forward(model: nn.Module, *, compile_config, ac_config) -> None:
    from torch.utils.checkpoint import checkpoint
    from torchtitan.distributed.activation_checkpoint import FullAC

    if ac_config is not None and not isinstance(ac_config, FullAC.Config):
        raise ValueError(f"--titan-fsdp-impl simple_fsdp supports full activation checkpointing only, got {ac_config}")
    blocks = list(model.layers.values())
    if compile_config.enable:
        backend = compile_config.backend
        for block in blocks:
            block.compile(backend=backend, fullgraph=True, dynamic=False if backend == "tpu" else None)
    if ac_config is not None:
        run_block = functools.partial(
            checkpoint,
            use_reentrant=False,
            preserve_rng_state=ac_config.preserve_rng_state,
            determinism_check=ac_config.determinism_check,
        )
    else:

        def run_block(block, *inputs):
            return block(*inputs)

    def forward(self, tokens, positions=None, attention_masks=None):
        # Decoder.forward, with each block run through run_block
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        for block in blocks:
            h = run_block(block, h, attention_masks, positions)
        h = self.norm(h) if self.norm is not None else h
        if self._skip_lm_head:
            return h
        return self.lm_head(h) if self.lm_head is not None else h

    model.forward = types.MethodType(forward, model)
    logger.info("Applied SimpleFSDP: collectives in each compiled TransformerBlock (spmd_safe)")
