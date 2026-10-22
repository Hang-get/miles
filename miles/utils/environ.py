import os
from collections.abc import Mapping
from pathlib import Path

_printed_legacy_rollout_v1 = False


def use_legacy_rollout_v1() -> bool:
    result = bool(int(os.environ.get("MILES_USE_LEGACY_ROLLOUT_V1", "0")))

    global _printed_legacy_rollout_v1
    if result and not _printed_legacy_rollout_v1:
        print("MILES_USE_LEGACY_ROLLOUT_V1=1 is enabled: using the deprecated v1 rollout path")
        _printed_legacy_rollout_v1 = True

    return result


def default_fp8_block_scaling_fp32_scales() -> str:
    """Default for NVTE_FP8_BLOCK_SCALING_FP32_SCALES, decided by hardware.

    On Blackwell (SM100+), TE emulates the blockwise FP8 recipe with MXFP8,
    which requires power-of-two scales, so FP32 scales must stay disabled.
    """
    import torch

    if not torch.cuda.is_available():
        return "1"
    major, _minor = torch.cuda.get_device_capability()
    return "0" if major >= 10 else "1"


def resolve_train_offload_disk_dir(directory: str | None, *, environment: Mapping[str, str] | None = None) -> Path:
    if directory is not None:
        return Path(directory)
    uid = os.getuid() if hasattr(os, "getuid") else 0  # config-access-exempt: os.getuid is platform-dependent
    environment = os.environ if environment is None else environment
    return Path(environment.get("SCRATCH", "/scratch")) / f"miles_train_offload_{uid}"
