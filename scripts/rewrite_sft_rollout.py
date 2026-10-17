"""Consume audited assistant-only masks, retaining the final partial batch."""

from math import ceil
from typing import Any


def generate_rollout(args: Any, rollout_id: int, data_buffer: Any, evaluation: bool = False) -> list:
    if evaluation:
        raise ValueError("Training-only SFT loader")
    size = len(data_buffer.dataset)
    batches = ceil(size / args.rollout_batch_size)
    count = min(args.rollout_batch_size, size - (rollout_id % batches) * args.rollout_batch_size)
    groups = data_buffer.get_samples(count)
    for group in groups:
        if len(group) != 1:
            raise ValueError("One sample per SFT conversation required")
        sample = group[0]
        tokens, mask = sample.metadata["sft_tokens"], sample.metadata["sft_mask"]
        if len(tokens) != len(mask) or not any(mask):
            raise ValueError("Invalid audited loss mask")
        sample.tokens = tokens
        sample.response_length = len(mask) - mask.index(1)
        sample.loss_mask = mask[-sample.response_length:]
        sample.reward = 0
    return groups
