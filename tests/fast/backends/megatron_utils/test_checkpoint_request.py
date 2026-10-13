import pytest
from miles.backends.megatron_utils.checkpoint_request import MegatronCkptLoadConfig
from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace


class TestCheckpointLoadScope:

    def test_nested_load_is_rejected_without_destroying_the_outer_request(self) -> None:
        """A nested checkpoint load cannot replace the active request or remove its fields."""
        args = MegatronArgsNamespace()
        outer = MegatronCkptLoadConfig.for_peer_recovery()
        inner = outer.model_copy(update={'load': 'other'})
        original = vars(args).copy()
        with outer.apply(args):
            with pytest.raises(AssertionError, match='runtime-only'):
                with inner.apply(args):
                    pytest.fail('Nested checkpoint scope unexpectedly entered')
            assert args.load is None
            assert args.no_load_optim is False
            assert args.no_load_rng is False
        assert vars(args) == original

    @pytest.mark.parametrize('field', ['load', 'finetune', 'no_load_optim', 'no_load_rng', 'ckpt_step', 'requested_load'])
    def test_preexisting_runtime_fields_are_rejected_without_mutation(self, field: str) -> None:
        """A loading request must not overwrite fields leaked from an earlier call."""
        args = MegatronArgsNamespace(**{field: 'existing'})
        original = vars(args).copy()
        with pytest.raises(AssertionError):
            with MegatronCkptLoadConfig.for_peer_recovery().apply(args):
                pytest.fail('Invalid checkpoint scope unexpectedly entered')
        assert vars(args) == original

    def test_sequential_requests_do_not_reuse_previous_load_inputs(self) -> None:
        """Each call receives its own checkpoint inputs and leaves the shared backend clean."""
        args = MegatronArgsNamespace()
        original = vars(args).copy()
        request = MegatronCkptLoadConfig.for_peer_recovery()
        with request.model_copy(update={'load': 'first', 'ckpt_step': 11}).apply(args):
            assert args.load == 'first'
            assert args.ckpt_step == 11
        with request.model_copy(update={'load': 'second', 'ckpt_step': 22}).apply(args):
            assert args.load == 'second'
            assert args.ckpt_step == 22
        assert vars(args) == original

    def test_requested_load_leaked_by_loader_is_rejected_on_exit(self) -> None:
        """A loader cannot silently introduce a second checkpoint source into the shared backend."""
        args = MegatronArgsNamespace()
        with pytest.raises(AssertionError, match='requested_load belongs to the orchestrator'):
            with MegatronCkptLoadConfig.for_peer_recovery().apply(args):
                vars(args)['requested_load'] = 'unexpected'
        assert not {'load', 'finetune', 'no_load_optim', 'no_load_rng', 'ckpt_step'} & vars(args).keys()
