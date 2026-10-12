import logging
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from radixscale.mesh import build_mesh
from radixscale.models.registry import build_model
from radixscale.packing import IGNORE_LABEL, CpShard, PackedBatch
from radixscale.spec import ParallelSpec
from radixscale.train import checkpoint
from radixscale.train.trainer import clip_grad_norm

from miles.backends.fsdp_utils.lr_scheduler import get_lr_scheduler
from miles.backends.radixscale_utils.hf_weight_iterator import RadixscaleHfWeightIterator
from miles.backends.radixscale_utils.parallel import create_radixscale_parallel_state
from miles.backends.training_utils.parallel import get_parallel_state, set_parallel_state
from miles.backends.training_utils.torch_native.actor import TorchNativeTrainRayActor
from miles.backends.training_utils.torch_native.step_runner import LinearStepRunner, StepMetrics
from miles.utils.context_utils import with_defer
from miles.utils.ft_utils.indep_dp import IndepDPInfo
from miles.utils.distributed_utils import get_gloo_group
from miles.utils.memory_utils import clear_memory, print_memory
from miles.utils.profile_utils import TrainProfiler
from miles.utils.timer import Timer, timer
from miles.utils.tracking_utils.tracking import init_tracking
from miles.utils.workers.rpc.common.wire_types import Pickled

logger = logging.getLogger(__name__)

_FAMILY_BY_MODEL_TYPE = (("glm", "glm"), ("qwen", "qwen"))


def _family(args, hf_config) -> str:
    if args.radixscale_family != "auto":
        return args.radixscale_family
    model_type = getattr(hf_config, "model_type", "")
    for prefix, family in _FAMILY_BY_MODEL_TYPE:
        if model_type.startswith(prefix):
            return family
    raise ValueError(f"radixscale has no model family for model_type {model_type!r}; pass --radixscale-family")


def _cp1_shard(batch: dict) -> CpShard:
    """The cp=1 shard built on the device from miles' packed microbatch: the host
    boundaries (cu_seqlens_host) size every op, so nothing waits on the GPU."""
    tokens = batch["tokens"][0]
    device = tokens.device
    bounds = batch["cu_seqlens_host"]
    lengths = torch.tensor([b - a for a, b in zip(bounds[:-1], bounds[1:])], dtype=torch.long)
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
        assert indep_dp_info.quorum_id == 0
        assert indep_dp_store_addr is None
        assert recv_ckpt_src_rank is None, "radixscale backend does not support checkpoint healing"
        assert not with_ref, "radixscale backend builds no reference model yet"
        assert not with_opd_teacher, "radixscale backend does not support on-policy distillation"

        world = dist.get_world_size()
        nvlink = int(os.environ.get("LOCAL_WORLD_SIZE", args.actor_num_gpus_per_node))
        # FSDP shards inside one NVLink domain and replicates across them (DESIGN.md D4)
        # unless a wider shard group is asked for explicitly
        fsdp = args.radixscale_fsdp_size or nvlink
        spec = ParallelSpec.create(
            world_size=world,
            nvlink_size=max(nvlink, fsdp),
            ep=args.expert_model_parallel_size,
            d_rep=max(world // fsdp, 1),
        )
        self.mesh = build_mesh(spec, "cuda")
        set_parallel_state(create_radixscale_parallel_state(self.mesh))
        if args.debug_rollout_only:
            return 0

        self.prof = TrainProfiler(args)
        self.load_hf_assets()
        built = build_model(
            _family(args, self.hf_config),
            Path(args.hf_checkpoint),
            self.mesh,
            recompute=args.gradient_checkpointing,
            compile=args.radixscale_compile,
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

    def _forward(self, batch: dict) -> torch.Tensor:
        return self.model(_cp1_shard(batch)).unsqueeze(0)

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
        """miles' losses assume FSDP's data-parallel mean. FSDP already averaged dense
        grads over the world; expert grads arrive summed over ep (tokens are dispatched
        to their owner) and averaged over e_fsdp only, and the fp32-critical params live
        outside FSDP (DESIGN.md N6)."""
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
        """Park parameter storage on the host by resizing it to zero, then refill it on
        wake: tensor objects and devices never change, which module._apply (swaps the
        DTensor params, refused while the profiler holds weakrefs) and param.data
        (cross-device rebinding is disallowed) cannot guarantee."""
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
        # release cache only when handing memory to the engine; no gc.collect: on a
        # trainer-sized heap it costs 1.2-1.6 s, and empty_cache itself ~1 s
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
        for storage, host in getattr(self, "_parked", []):
            storage.resize_(host.numel())
            torch.empty(0, dtype=torch.uint8, device="cuda").set_(storage).copy_(host, non_blocking=True)
        torch.cuda.synchronize()
        self._parked = []

    @timer
    def update_weights(self, info):  # type: ignore[override]
        """Colocated syncs run while the trainer sleeps: bring the parameters (not the
        optimizer state) back for the gathers, then park them again. No gc.collect
        afterwards (the base class's clear_memory costs ~1.3 s on this heap)."""
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return None
        parked = bool(getattr(self, "_parked", None))
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
