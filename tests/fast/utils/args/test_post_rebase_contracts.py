import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config
from miles.utils.args.runtime import TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestNativeCheckpointStateFlags:

    @pytest.mark.parametrize('flag,field', [('--no-load-optim', 'no_load_optim'), ('--no-load-rng', 'no_load_rng'), ('--no-save-optim', 'no_save_optim')])
    def test_checkpoint_flags_reach_the_fsdp_backend_without_flat_duplicates(self, flag: str, field: str) -> None:
        """Shared checkpoint flags survive FSDP parsing and trainer payload construction."""
        config = parse_fsdp_test_config(flag)
        trainer = compute_trainer_config(config, config.raw_megatron.trainers[0])
        assert vars(trainer.backend)[field] is True
        assert field not in TrainerConfig.model_fields
        restored = TrainerConfig.model_validate_json(trainer.model_dump_json())
        assert vars(restored.backend)[field] is True
