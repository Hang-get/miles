from tests.fast.fixtures.args_fixtures import parse_megatron_test_config
from miles.utils.args.runtime import OrchestratorConfig
from miles.utils.args.trainer_utils import compute_trainer_config


class TestOrchestratorCheckpointRequests:

    def test_requested_load_is_not_transported_to_trainers(self) -> None:
        """The requested destination remains with orchestration while trainers receive loading inputs per call."""
        args = parse_megatron_test_config('--load', '/run', '--ref-load', '/reference')
        orchestrator = OrchestratorConfig.slice_from(args)
        trainer = compute_trainer_config(args, args.raw_megatron.trainers[0])
        assert orchestrator.requested_load == '/run'
        assert 'requested_load' not in trainer.model_dump()
        assert 'requested_load' not in vars(trainer.backend)
