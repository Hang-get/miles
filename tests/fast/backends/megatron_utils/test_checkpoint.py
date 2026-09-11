from argparse import Namespace
from collections.abc import Callable
from contextlib import nullcontext

import pytest
import torch

from miles.backends.megatron_utils import checkpoint
from miles.backends.training_utils.model_companion import ModelCompanion


class TestRawMbridgeCompanionIsolation:
    @pytest.mark.parametrize("fail", [False, True])
    def test_hf_loading_excludes_and_restores_companion_state(
        self, raw_mbridge_loader: Callable[[bool], list[set[str]]], fail: bool
    ) -> None:
        """HF loading must not map training metadata or discard it when loading fails."""
        observed = raw_mbridge_loader(fail)
        models = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        companions = [ModelCompanion(pipeline_rank=0, chunk_index=i, replica_id=(0, 0, 0)) for i in range(2)]
        for model, companion in zip(models, companions, strict=True):
            companion.weight_version.fill_(7)
            companion.sample_consumptions.data = torch.tensor([[1, 2, 3, 4, 0]])
            model.add_module("model_companion", companion)

        with pytest.raises(RuntimeError, match="HF loading failed") if fail else nullcontext():
            checkpoint._load_hf_weights_with_mbridge(
                ddp_model=models, args=Namespace(offload_train=False), load_path="/unused/hf"
            )

        assert observed == [{"weight", "bias"}, {"weight", "bias"}]
        for model, companion in zip(models, companions, strict=True):
            assert model.model_companion is companion
            assert companion.weight_version.item() == 7
            assert companion.sample_consumptions.tolist() == [[1, 2, 3, 4, 0]]
            torch.testing.assert_close(model.weight, torch.full_like(model.weight, 5))
