"""Lightweight checks for exact SFT epoch coverage and audited-mask handling."""

import unittest
from types import SimpleNamespace

from scripts.rewrite_sft_rollout import generate_rollout


class DataBuffer:
    def __init__(self, size: int) -> None:
        self.dataset = range(size)
        self.requested = []

    def get_samples(self, count: int) -> list:
        self.requested.append(count)
        return [[SimpleNamespace(metadata={"sft_tokens": [1, 2, 3, 4], "sft_mask": [0, 0, 1, 1]})]
                for _ in range(count)]


class RewriteSftTest(unittest.TestCase):
    def test_exact_epoch_with_partial_final_batch(self) -> None:
        args = SimpleNamespace(rollout_batch_size=4)
        buffer = DataBuffer(10083)
        for rollout_id in range(2521):
            samples = generate_rollout(args, rollout_id, buffer)
            self.assertTrue(all(s[0].response_length == 2 and s[0].loss_mask == [1, 1] for s in samples))
        self.assertEqual(sum(buffer.requested), 10083)
        self.assertEqual(buffer.requested[-1], 3)
        self.assertEqual(len(generate_rollout(args, 2521, buffer)), 4)

    def test_reject_evaluation(self) -> None:
        with self.assertRaises(ValueError):
            generate_rollout(SimpleNamespace(rollout_batch_size=4), 0, DataBuffer(3), evaluation=True)


if __name__ == "__main__":
    unittest.main()
