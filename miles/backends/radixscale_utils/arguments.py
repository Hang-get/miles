from dataclasses import dataclass

from miles.backends.fsdp_utils.arguments import FSDPArgs, build_dataclass_parser, load_args_from_parser


@dataclass
class RadixscaleArgs(FSDPArgs):
    # radixscale.models.registry family; "auto" picks it from the HF config's model_type
    radixscale_family: str = "auto"
    radixscale_compile: bool = False
    # with --gradient-checkpointing: keep matmul/attention outputs, recompute only the cheap rest
    radixscale_selective_recompute: bool = False
    # compute per-token log-probs/entropy chunk by chunk inside the model instead of returning [T, vocab] logits
    radixscale_fused_logprobs: bool = True
    # with --gradient-checkpointing: the last N layers keep their activations (no recompute)
    radixscale_recompute_skip_layers: int = 0
    # dense FSDP shard group size; 0 = one NVLink domain (replicas across domains)
    radixscale_fsdp_size: int = 0
    # expert replicas across NVLink domains (expert HSDP); 0 shards experts over the whole world
    radixscale_expert_rep: int = 0
    expert_model_parallel_size: int = 1


def build_radixscale_parser(extra_args_provider=None):
    return build_dataclass_parser(RadixscaleArgs, "radixscale Training (miles)", extra_args_provider)


def load_radixscale_args(extra_args_provider=None):
    return load_args_from_parser(build_radixscale_parser(extra_args_provider))


def validate_radixscale_args(args) -> None:
    if args.context_parallel_size != 1:
        raise ValueError("the radixscale backend runs context parallelism through its own packing, not wired up yet")
    if args.use_kl_loss or args.kl_coef:
        raise ValueError("the radixscale backend builds no reference model yet; drop --use-kl-loss / --kl-coef")
    if args.fp16:
        raise ValueError("the radixscale backend trains in bf16 mixed precision; --fp16 is not supported")
    if args.lr_decay_style not in ("constant", "linear", "cosine"):
        raise ValueError(f"unsupported --lr-decay-style {args.lr_decay_style!r}; use constant, linear or cosine")
    if args.qkv_format != "thd":
        raise ValueError("the radixscale backend consumes packed thd microbatches only")
