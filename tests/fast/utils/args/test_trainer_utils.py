from pathlib import Path

import pytest
from tests.fast.fixtures.args_fixtures import parse_fsdp_test_config, parse_megatron_test_config
from tests.fast.fixtures.megatron_config_fixtures import encode_megatron_config

from miles.backends.megatron_utils.megatron_config import MegatronArgsNamespace
from miles.utils.args.runtime import OrchestratorConfig, TrainerConfig
from miles.utils.args.trainer_utils import compute_trainer_ckpt_load_config, compute_trainer_config
from miles.utils.run_uuid import RUN_UUID_LENGTH


class TestTrainerCheckpointConfiguration:
    def test_checkpoint_creation_preserves_the_complete_trainer_config(self, tmp_path: Path) -> None:
        """Identical launch arguments must keep trainer Pod configuration unchanged after saving."""
        load = tmp_path / "run"
        argv = ("--load", str(load), "--ref-load", str(tmp_path / "reference"), "--run-uuid", "a" * RUN_UUID_LENGTH)
        before = parse_megatron_test_config(*argv)
        [trainer] = before.raw_megatron.trainers
        original = compute_trainer_config(before, trainer).model_dump(mode="json")

        load.mkdir()
        (load / "latest_checkpointed_iteration.txt").write_text("3")
        after = parse_megatron_test_config(*argv)
        [trainer] = after.raw_megatron.trainers
        repeated = compute_trainer_config(after, trainer).model_dump(mode="json")

        assert original == repeated

    @pytest.mark.parametrize("field", ["load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step"])
    def test_runtime_loading_fields_are_rejected_in_trainer_configuration(self, field: str) -> None:
        """A serialized loading decision must not silently become part of a trainer Pod template."""
        args = parse_megatron_test_config()
        [trainer] = args.raw_megatron.trainers
        config = compute_trainer_config(args, trainer)
        assert field not in vars(config.backend)
        values = config.model_dump(mode="python")
        values["backend"] = MegatronArgsNamespace(**(vars(config.backend) | {field: None}))
        with pytest.raises(ValueError, match="Checkpoint load inputs"):
            TrainerConfig.model_validate(values)


class TestOrchestratorCheckpointRequests:
    @pytest.mark.parametrize(
        "extra",
        [(), ("--advantage-estimator", "ppo"), ("--megatron-config", encode_megatron_config("a", "b"))],
    )
    def test_loading_uses_only_fields_owned_by_the_orchestrator(self, extra: tuple[str, ...]) -> None:
        """Checkpoint selection must work after deployment-only scaling fields have been removed."""
        args = parse_megatron_test_config(*extra, "--load", "/run", "--ref-load", "/reference")
        orchestrator = OrchestratorConfig.slice_from(args)

        assert "actor_num_nodes" not in type(orchestrator).model_fields
        for trainer in args.raw_megatron.trainers:
            request = compute_trainer_ckpt_load_config(orchestrator, trainer)
            assert request == compute_trainer_ckpt_load_config(args, trainer)
            assert request.load == "/reference"

    def test_requested_load_is_not_transported_to_trainers(self) -> None:
        """The requested destination remains with orchestration while trainers receive loading inputs per call."""
        args = parse_megatron_test_config("--load", "/run", "--ref-load", "/reference")
        orchestrator = OrchestratorConfig.slice_from(args)
        trainer = compute_trainer_config(args, args.raw_megatron.trainers[0])

        assert orchestrator.requested_load == "/run"
        assert "requested_load" not in trainer.model_dump()
        assert "requested_load" not in vars(trainer.backend)


class TestSinglePolicyCriticCheckpointRequests:
    @pytest.mark.parametrize("critic_source", ["unset", "hf", "tracked", "same_as_actor"])
    def test_fresh_actor_controls_flags_without_replacing_explicit_critic_weights(
        self, tmp_path: Path, critic_source: str
    ) -> None:
        """Fresh PPO keeps explicit critic weights while inheriting the actor's fresh-load flags."""
        actor = tmp_path / "actor"
        reference = tmp_path / "reference"
        critic = actor if critic_source == "same_as_actor" else tmp_path / "critic"
        critic.mkdir()
        if critic_source == "tracked":
            (critic / "latest_checkpointed_iteration.txt").write_text("7")
        extra = [] if critic_source == "unset" else ["--critic-load", str(critic)]
        args = parse_megatron_test_config(
            "--advantage-estimator", "ppo", "--load", str(actor), "--ref-load", str(reference), *extra
        )
        orchestrator = OrchestratorConfig.slice_from(args)
        [actor_config, critic_config] = args.raw_megatron.trainers
        before = orchestrator.model_dump(mode="python")

        actor_request = compute_trainer_ckpt_load_config(orchestrator, actor_config)
        critic_request = compute_trainer_ckpt_load_config(orchestrator, critic_config)

        assert actor_request.load == str(reference)
        assert critic_request.load == str(reference if critic_source == "unset" else critic)
        assert critic_request.resume_from_ckpt is False
        assert critic_request.finetune is True
        assert critic_request.no_load_optim is True
        assert critic_request.no_load_rng is True
        assert critic_request.model_dump(exclude={"load"}) == actor_request.model_dump(exclude={"load"})
        assert orchestrator.model_dump(mode="python") == before
        assert args.critic_load == (None if critic_source == "unset" else str(critic))
        trainer = compute_trainer_config(args, critic_config)
        assert not {"load", "finetune", "no_load_optim", "no_load_rng", "ckpt_step"} & vars(trainer.backend).keys()

    def test_resuming_actor_keeps_critic_checkpoint_and_optimizer_restore_flags(self, tmp_path: Path) -> None:
        """A normal PPO resume preserves each checkpoint path and the actor's optimizer restore policy."""
        actor = tmp_path / "actor"
        critic = tmp_path / "critic"
        for directory in (actor, critic):
            directory.mkdir()
            (directory / "latest_checkpointed_iteration.txt").write_text("7")
        args = parse_megatron_test_config(
            "--advantage-estimator", "ppo", "--load", str(actor), "--critic-load", str(critic)
        )
        [actor_config, critic_config] = args.raw_megatron.trainers

        actor_request = compute_trainer_ckpt_load_config(args, actor_config)
        critic_request = compute_trainer_ckpt_load_config(args, critic_config)

        assert actor_request.load == str(actor)
        assert critic_request.load == str(critic)
        assert critic_request.resume_from_ckpt is True
        assert not critic_request.finetune
        assert not critic_request.no_load_optim
        assert not critic_request.no_load_rng
        assert critic_request.model_dump(exclude={"load"}) == actor_request.model_dump(exclude={"load"})

    def test_named_critic_still_resolves_its_own_checkpoint_independently(self, tmp_path: Path) -> None:
        """Named trainer checkpoint namespaces retain their existing per-trainer resolution."""
        critic = tmp_path / "critic"
        directory = critic / "trainers" / "policy-critic"
        directory.mkdir(parents=True)
        (directory / "latest_checkpointed_iteration.txt").write_text("7")
        args = parse_megatron_test_config(
            "--advantage-estimator",
            "ppo",
            "--megatron-config",
            encode_megatron_config("policy"),
            "--load",
            str(tmp_path / "actor"),
            "--ref-load",
            str(tmp_path / "reference"),
            "--critic-load",
            str(critic),
        )
        [actor_config, critic_config] = args.raw_megatron.trainers

        actor_request = compute_trainer_ckpt_load_config(args, actor_config)
        critic_request = compute_trainer_ckpt_load_config(args, critic_config)

        assert actor_request.resume_from_ckpt is False
        assert critic_request.resume_from_ckpt is True
        assert critic_request.load == str(directory)


class TestCriticLoadDefaultBoundaries:
    def test_non_megatron_default_still_uses_the_resolved_actor_source(self, tmp_path: Path) -> None:
        """Non-Megatron parsing keeps its existing eager critic default."""
        reference = str(tmp_path / "reference")
        args = parse_fsdp_test_config("--load", str(tmp_path / "missing"), "--ref-load", reference)

        assert args.critic_load == reference

    def test_named_megatron_default_still_uses_the_requested_actor_source(self, tmp_path: Path) -> None:
        """Named trainer parsing retains its existing critic default and directory resolution."""
        actor = str(tmp_path / "actor")
        args = parse_megatron_test_config("--megatron-config", encode_megatron_config("policy"), "--load", actor)

        assert args.critic_load == actor
