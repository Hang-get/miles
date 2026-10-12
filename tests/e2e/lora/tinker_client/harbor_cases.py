import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from tinker_agent import harbor
from tinker_agent.train import HarborTinkerConfig, build_config
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.rl.data_processing import trajectory_to_data

TEMPLATE = json.dumps(
    {
        "environment": {"type": "docker"},
        "agent": {
            "name": "terminus-2",
            "model_name": "openai/policy",
            "kwargs": {
                "api_base": "{proxy_url}/v1",
                "llm_kwargs": {"api_key": "{proxy_api_key}"},
            },
        },
    }
)


@pytest.mark.asyncio
async def test_harbor_http_calls_become_unmodified_training_data(monkeypatch, renderer, sampler_factory):
    sampler = sampler_factory()
    sampler.get_tokenizer = lambda: renderer.tokenizer
    policy = TinkerTokenCompleter(sampler, max_tokens=32)
    configs = []

    async def create(config):
        configs.append(config)
        reward = float(len(configs) % 2)

        async def run():
            async with httpx.AsyncClient() as http:
                for text in ["original", "compacted", "compacted"]:
                    response = await http.post(
                        f"{config.agent.kwargs['api_base']}/chat/completions",
                        headers={"Authorization": f"Bearer {config.agent.kwargs['llm_kwargs']['api_key']}"},
                        json={"messages": [{"role": "user", "content": text}]},
                    )
                    response.raise_for_status()
            return SimpleNamespace(exception_info=None, verifier_result=SimpleNamespace(rewards={"reward": reward}))

        return SimpleNamespace(run=run)

    monkeypatch.setattr(harbor.Trial, "create", create)
    strategy = harbor.HarborRolloutStrategy("role_colon", TEMPLATE, max_parallel_trials_per_group=1)
    strategy = pickle.loads(pickle.dumps(strategy))
    group = harbor.HarborGroup(Path("/tasks/task"), 2)
    result = await strategy.execute(group, policy)
    assert len({c.agent.kwargs["api_base"] for c in configs}) == 2
    assert all(c.task.path == Path("/tasks/task") for c in configs)
    assert [reward for reward, _ in await group.compute_group_rewards(result.trajectories, result.envs)] == [1, 0]
    for trajectory in result.trajectories:
        datums = trajectory_to_data(trajectory, traj_advantage=1.0)
        assert len(datums) == 3
        assert [datum.loss_fn_inputs["target_tokens"].data[-1] for datum in datums] == [1000] * 3
        assert [datum.loss_fn_inputs["logprobs"].data[-1] for datum in datums] == [-0.25] * 3
    async with httpx.AsyncClient() as http:
        with pytest.raises(httpx.ConnectError):
            await http.get(configs[0].agent.kwargs["api_base"])


@pytest.mark.parametrize("failure,reason", [(None, "completed"), ("AgentTimeoutError", "rollout_timeout")])
def test_only_scored_trial_outcomes_are_accepted(failure, reason):
    result = SimpleNamespace(
        exception_info=None if failure is None else SimpleNamespace(exception_type=failure),
        verifier_result=SimpleNamespace(rewards={"reward": 0.5}),
    )
    assert harbor.trial_outcome(result, "reward") == (0.5, reason)


@pytest.mark.parametrize(
    "failure,rewards",
    [
        ("VerifierTimeoutError", {"reward": 0}),
        ("NonZeroAgentExitCodeError", {"reward": 0}),
        (None, None),
        ("AgentTimeoutError", None),
        (None, {"reward": float("nan")}),
    ],
)
def test_failed_or_unscored_trials_cannot_turn_into_zero_rewards(failure, rewards):
    result = SimpleNamespace(
        exception_info=(
            None if failure is None else SimpleNamespace(exception_type=failure, exception_message="failed")
        ),
        verifier_result=SimpleNamespace(rewards=rewards),
    )
    with pytest.raises((RuntimeError, ValueError)):
        harbor.trial_outcome(result, "reward")


def test_recipe_uses_public_sdk_configuration_and_harbor_defaults(tmp_path):
    config_path = tmp_path / "trial.json"
    config_path.write_text(TEMPLATE)
    config = build_config(
        HarborTinkerConfig(
            model_name="model",
            renderer_name="role_colon",
            tasks_dir=str(tmp_path),
            trial_config=str(config_path),
        )
    )
    assert config.base_url is None
    assert config.ttl_seconds is None
    trial = harbor.trial_config(TEMPLATE, tmp_path, "http://proxy/r/traj/7", 'key"quoted')
    assert trial.agent.kwargs["api_base"] == "http://proxy/r/traj/7/v1"
    assert trial.agent.kwargs["llm_kwargs"]["api_key"] == 'key"quoted'
    assert "response_length_exceeded_policy" not in trial.agent.kwargs
    assert "max_turns" not in trial.agent.kwargs
