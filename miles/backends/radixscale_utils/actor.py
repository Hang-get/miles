import os
from pathlib import Path

import torch
import torch.distributed as dist
from radixscale.mesh import build_mesh
from radixscale.models.registry import build_model
from radixscale.packing import IGNORE_LABEL, CpShard, PackedBatch
from radixscale.spec import ParallelSpec
from radixscale.train import checkpoint
from radixscale.train.trainer import clip_grad_norm
from torch.distributed.tensor import DTensor

from miles.backends.fsdp_utils.lr_scheduler import get_lr_scheduler
from miles.backends.radixscale_utils.hf_weight_iterator import RadixscaleHfWeightIterator
from miles.backends.radixscale_utils.parallel import create_radixscale_parallel_state
from miles.backends.training_utils.loss.hub.logit_processors import TokenLogProbs
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native.step_runner import LinearStepRunner, StepMetrics
from miles.ray.rollout.inference_controller import UpdatableEngines
from miles.utils.context_utils import with_defer
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer, timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.workers.rpc.common.wire_types import Pickled

_FAMILY_BY_MODEL_TYPE = (("glm", "glm"), ("qwen", "qwen"))


def _family(requested: str, model_type: str) -> str:
    if requested != "auto":
        return requested
    for prefix, family in _FAMILY_BY_MODEL_TYPE:
        if model_type.startswith(prefix):
            return family
    raise ValueError(f"radixscale has no model family for model_type {model_type!r}; pass --radixscale-family")


def _cp1_shard(batch: dict) -> CpShard:
    tokens = batch["tokens"][0]
    device = tokens.device
    bounds = batch["cu_seqlens_host"]
    lengths = torch.tensor([b - a for a, b in zip(bounds[:-1], bounds[1:], strict=True)], dtype=torch.long)
    seq_ids = torch.repeat_interleave(torch.arange(len(lengths)), lengths).to(device, non_blocking=True)
    cu_seqlens = torch.tensor(bounds, dtype=torch.int32).to(device, non_blocking=True)
    empty = torch.zeros(0, dtype=torch.long, device=device)
    pack = PackedBatch(
        input_ids=tokens,
        labels=torch.full_like(tokens, IGNORE_LABEL),
        position_ids=batch["position_ids"][0],
        cu_seqlens=cu_seqlens,
        max_seqlen=int(batch["max_seqlen"]),
    )
    return CpShard(
        local=pack,
        cp_rank=0,
        cp_size=1,
        global_cu_seqlens=cu_seqlens,
        global_max_seqlen=pack.max_seqlen,
        gather_index=empty,
        local_seq_ids=seq_ids,
        cp_kv_rows=empty,
        cp_cu_k=empty.to(torch.int32),
        cp_max_k=0,
    )


def _next_token_labels(batch: dict) -> torch.Tensor:
    tokens = batch["tokens"][0]
    sample_ends = torch.tensor(batch["cu_seqlens_host"][1:], dtype=torch.long) - 1
    labels = torch.roll(tokens, -1)
    sample_ends = sample_ends.clamp(min=0).pin_memory() if tokens.is_cuda else sample_ends.clamp(min=0)
    return labels.index_fill_(0, sample_ends.to(tokens.device, non_blocking=True), IGNORE_LABEL)


def _fused_logprobs_supported(args) -> bool:
    return (
        args.radixscale_fused_logprobs
        and not args.true_on_policy_mode
        and not args.use_sampling_support_replay
        and args.loss_type == "policy_loss"
        and args.context_parallel_size == 1
        and args.qkv_format == "thd"
    )


class RadixscaleTrainRayActor(TorchNativeTrainRayActor):
    backend_name = "radixscale"

    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Pickled,
        role: str,
        *,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
        recv_ckpt_src_rank: int | None = None,
        indep_dp_info: IndepDPInfo,
        indep_dp_store_addr: str | None,
    ) -> int | None:  # type: ignore[override]
        super()._init_common(args, role, with_ref, with_opd_teacher=with_opd_teacher)
        self._parked: list[tuple[torch.UntypedStorage, torch.Tensor]] = []
        assert indep_dp_info.quorum_id == 0
        assert indep_dp_store_addr is None
        assert recv_ckpt_src_rank is None, "radixscale backend does not support checkpoint healing"
        assert not with_ref, "radixscale backend builds no reference model yet"
        assert not with_opd_teacher, "radixscale backend does not support on-policy distillation"

        world = dist.get_world_size()
        nvlink = int(os.environ.get("LOCAL_WORLD_SIZE", args.actor_num_gpus_per_node))
        fsdp = args.radixscale_fsdp_size or nvlink
        spec = ParallelSpec.create(
            world_size=world,
            nvlink_size=max(nvlink, fsdp),
            ep=args.expert_model_parallel_size,
            d_rep=max(world // fsdp, 1),
            e_rep=args.radixscale_expert_rep or 1,
        )
        self.mesh = build_mesh(spec, "cuda")
        set_parallel_state(create_radixscale_parallel_state(self.mesh))
        if args.debug_rollout_only:
            return 0

        self._fused_logprobs = _fused_logprobs_supported(args)
        self.prof = TrainProfiler(args)
        self.load_hf_assets()
        built = build_model(
            _family(args.radixscale_family, self.hf_config.model_type),
            Path(args.hf_checkpoint),
            self.mesh,
            recompute=args.gradient_checkpointing,
            recompute_skip_last_layers=args.radixscale_recompute_skip_layers,
        )
        self.model = built.model
        self.grad_sum_factors = built.grad_sum_factors
        self.model_parts = [self.model]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            fused=True,
        )
        self.optimizers = [self.optimizer]
        self.lr_scheduler = get_lr_scheduler(args, self.optimizer)

        if get_parallel_state().is_metrics_rank:
            init_tracking(args, primary=False)

        start_rollout_id = 0
        if args.load:
            latest = checkpoint.latest_step(Path(args.load))
            if latest is not None:
                checkpoint.load_step(Path(args.load), latest, self.model, self.optimizer)
                start_rollout_id = latest

        self.weight_updater = self._build_weight_updater(self.model, RadixscaleHfWeightIterator.build)

        clear_memory()
        if args.offload_train:
            self.sleep()
        self.prof.on_init_end()
        return args.start_rollout_id if args.start_rollout_id is not None else start_rollout_id

    def _step_runner(self) -> LinearStepRunner:
        return LinearStepRunner(self._forward, self._zero_grad, self._apply_step)

    def _forward(self, batch: dict) -> torch.Tensor | TokenLogProbs:
        shard = _cp1_shard(batch)
        if not self._fused_logprobs:
            return self.model(shard).unsqueeze(0)
        args = self.args
        with_entropy = not torch.is_grad_enabled() or args.entropy_coef != 0 or args.observe_training_entropy
        out = self.model(
            shard,
            logprob_labels=_next_token_labels(batch),
            temperature=args.rollout_temperature if args.rollout_temperature > 0 else 1.0,
            with_entropy=with_entropy,
        )
        log_probs, entropy = out if with_entropy else (out, None)
        return TokenLogProbs(log_probs.unsqueeze(0), None if entropy is None else entropy.unsqueeze(0))

    def _zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _apply_step(self) -> StepMetrics:
        self._mean_reduce_grads()
        grad_norm = clip_grad_norm(self.model, self.args.clip_grad)
        self.optimizer.step()
        self.lr_scheduler.step()
        return StepMetrics(
            grad_norm=float(grad_norm.item()),
            extra_metrics={f"lr-pg_{i}": group["lr"] for i, group in enumerate(self.optimizer.param_groups)},
        )

    def _mean_reduce_grads(self) -> None:
        """Bring every grad to the data-parallel mean miles' losses assume."""
        world = dist.get_world_size()
        ep = self.mesh.spec.ep
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            factor = self.grad_sum_factors.get(name, world) if self.grad_sum_factors else world
            if factor is None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad.div_(world)
            elif ".experts." in name and ep > 1:
                param.grad.div_(ep)

    def _move_to(self, device: str) -> None:
        print_memory(f"before moving the model to {device}")
        with torch.no_grad():
            if device == "cpu":
                self.optimizer.zero_grad(set_to_none=True)
                self._park_params()
            else:
                self._unpark_params()
            for state in self.optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device, non_blocking=True)
        torch.cuda.synchronize()
        if device == "cpu":
            torch.cuda.empty_cache()
        dist.barrier(group=get_gloo_group())
        print_memory(f"after moving the model to {device}")

    def _park_params(self) -> None:
        self._parked = []
        seen = set()
        for param in self.model.parameters():
            local = param._local_tensor if isinstance(param, DTensor) else param.data
            storage = local.untyped_storage()
            if storage.data_ptr() in seen or storage.nbytes() == 0:
                continue
            seen.add(storage.data_ptr())
            view = torch.empty(0, dtype=torch.uint8, device=local.device).set_(storage)
            host = torch.empty(storage.nbytes(), dtype=torch.uint8, pin_memory=True)
            host.copy_(view, non_blocking=True)
            self._parked.append((storage, host))
        torch.cuda.synchronize()
        for storage, _host in self._parked:
            storage.resize_(0)

    def _unpark_params(self) -> None:
        for storage, host in self._parked:
            storage.resize_(host.numel())
            torch.empty(0, dtype=torch.uint8, device="cuda").set_(storage).copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        self._parked = []

    @timer
    def update_weights(self, info: UpdatableEngines) -> int | None:  # type: ignore[override]
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return None
        parked = bool(self._parked)
        with torch.no_grad():
            if parked:
                self._unpark_params()
            try:
                self.weight_updater.reconnect_if_needed(info)
                print_memory("before update_weights")
                self.weight_updater.update_weights()
                print_memory("after update_weights")
                if self.args.ci_test:
                    self.weight_updater.verify_engine_version(info.rollout_engines)
            finally:
                if parked:
                    self._park_params()
                    torch.cuda.empty_cache()
        return self.weight_updater.weight_version

    def _save_checkpoint(self, rollout_id: int) -> None:
        checkpoint.save_step(Path(self.args.save), rollout_id + 1, self.model, self.optimizer)
