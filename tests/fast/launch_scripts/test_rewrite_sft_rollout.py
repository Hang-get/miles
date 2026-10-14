from types import SimpleNamespace

from scripts.rewrite_sft_rollout import generate_rollout


class Buffer:
    def __init__(self, size: int) -> None:
        self.dataset = range(size)
        self.offset = 0
        self.seen = []

    def get_samples(self, count: int) -> list:
        assert self.offset + count <= len(self.dataset)
        rows = []
        for index in range(self.offset, self.offset + count):
            self.seen.append(index)
            rows.append([SimpleNamespace(metadata={"sft_tokens": [5, 6, 7], "sft_mask": [0, 1, 1]})])
        self.offset = (self.offset + count) % len(self.dataset)
        return rows


def test_two_exact_epochs_with_partial_batches() -> None:
    buffer = Buffer(3870)
    args = SimpleNamespace(rollout_batch_size=4)
    sizes = []
    for rollout in range(1936):
        groups = generate_rollout(args, rollout, buffer)
        sizes.append(len(groups))
        for group in groups:
            sample = group[0]
            assert sample.tokens == [5, 6, 7]
            assert sample.response_length == 2
            assert sample.loss_mask == [1, 1]
    assert sizes.count(2) == 2
    assert sizes[967] == sizes[1935] == 2
    assert buffer.seen == list(range(3870)) * 2
