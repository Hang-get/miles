import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainDumpMetadata:
    weight_version: int
    attempt: int
    cell_index: int


def save_debug_train_data(
    args: Any, *, rollout_id: int, rollout_data: dict[str, Any], metadata: TrainDumpMetadata | None = None
) -> None:
    if args.save_debug_train_data is not None:
        parallel_state = get_parallel_state()
        # TP peers duplicate the DP shard and only the last PP stage computes the per-token fields
        if parallel_state.tp.rank != 0 or not parallel_state.is_pp_last_stage:
            return
        save_debug_train_data_for_rank(
            args,
            rollout_id=rollout_id,
            rollout_data=rollout_data,
            rank=torch.distributed.get_rank(),
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
            metadata=metadata,
        )


def save_debug_train_data_for_rank(
    args: Any,
    *,
    rollout_id: int,
    rollout_data: dict[str, Any],
    rank: int,
    cp_rank: int = 0,
    cp_size: int = 1,
    metadata: TrainDumpMetadata | None = None,
) -> None:
    """Write one rank's slice of a rollout.

    ``cp_rank`` / ``cp_size`` are passed in for the same reason ``rank`` is:
    this runs outside the process group in tests and offline tools.
    """
    if (path_template := args.save_debug_train_data) is not None:
        fields = asdict(metadata) if metadata is not None else {}
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank, **fields))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
                cp_rank=cp_rank,
                cp_size=cp_size,
                qkv_format=args.qkv_format,
                metadata=fields,
            ),
            path,
        )
