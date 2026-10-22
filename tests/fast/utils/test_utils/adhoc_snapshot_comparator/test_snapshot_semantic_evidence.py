from argparse import ArgumentParser, Namespace
from collections.abc import Callable

import pytest
import torch

from miles.backends.training_utils.loss.hub.corrections import icepop_function, vanilla_tis_function
from miles.backends.training_utils.loss.hub.math_utils import get_reinforce_plus_plus_baseline_advantages
from miles.utils.args.custom_function import CustomFunctionConfig, resolve_custom_function_configs
from miles.utils.args.schema import BaseConfig
from miles.utils.arguments import _validate_argument_ownership


class _HookSelection(BaseConfig):
    custom_generate_function_path: CustomFunctionConfig


class TestSnapshotSemanticEvidence:
    @pytest.mark.parametrize("function", [icepop_function, vanilla_tis_function])
    def test_zero_clip_bound_type_change_preserves_real_loss_outputs(
        self, function: Callable[..., tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]]
    ) -> None:
        """The unchanged production consumers compute identical tensors for captured 0 and 0.0 bounds."""
        results = [
            function(
                Namespace(tis_clip_low=value, tis_clip=2.0),
                pg_loss=torch.tensor([1.0, -2.0, 3.0]),
                train_log_probs=[torch.tensor([-20.0, 0.0, 20.0])],
                rollout_log_probs=[torch.zeros(3)],
                loss_masks=[torch.ones(3)],
            )
            for value in (0, 0.0)
        ]
        assert torch.equal(results[0][0], results[1][0])
        assert all(torch.equal(results[0][2][key], results[1][2][key]) for key in results[0][2])

    def test_zero_kl_type_change_preserves_the_production_reward_expression(self) -> None:
        """Captured critic 0 and 0.0 coefficients both disable the unchanged KL contribution."""
        outputs = [
            get_reinforce_plus_plus_baseline_advantages(
                rewards=torch.tensor([2.0]),
                kl=[torch.tensor([-3.0, 0.0, 4.0])],
                loss_masks=[torch.ones(3)],
                kl_coef=value,
            )
            for value in (0, 0.0)
        ]
        assert torch.equal(outputs[0][0], outputs[1][0])

    @pytest.mark.parametrize("value", ["invalid_snapshot_path", "miles_snapshot_missing.module"])
    def test_removed_invalid_hook_case_is_rejected_by_final_resolution(self, value: str) -> None:
        """The removed successful parser fixtures are invalid at the actual configuration boundary."""
        with pytest.raises((ValueError, ModuleNotFoundError)):
            resolve_custom_function_configs(
                Namespace(custom_generate_function_path=value), config_class=_HookSelection
            )

    @pytest.mark.parametrize("backend", ["fsdp", "megatron"])
    def test_removed_callback_field_is_rejected_by_the_real_owner_check(self, backend: str) -> None:
        """Bare callback fields have no owner under the human-approved strict configuration contract."""
        with pytest.raises(ValueError, match="snapshot_custom"):
            _validate_argument_ownership(
                Namespace(train_backend=backend, snapshot_custom=29),
                parser=ArgumentParser(),
                training_backend_arg_names=set(),
            )
