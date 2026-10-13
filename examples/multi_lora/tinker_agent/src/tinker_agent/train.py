import asyncio
from pathlib import Path
from uuid import uuid4

import chz
from tinker_agent.harbor import HarborDatasetBuilder, HarborRolloutStrategy
from tinker_cookbook.capture.store import capture_to_store
from tinker_cookbook.rl import train


@chz.chz
class HarborTinkerConfig:
    model_name: str
    renderer_name: str
    tasks_dir: str
    trial_config: str
    base_url: str | None = None
    log_path: str = "/tmp/harbor-tinker"
    reward_key: str = "reward"
    lora_rank: int = 16
    groups_per_batch: int = 4
    group_size: int = 4
    epochs: int = 1
    advertised_host: str = "127.0.0.1"
    listen_host: str = "127.0.0.1"
    max_parallel_trials_per_group: int = 4
    max_tokens: int = 8192
    loss_fn: str = "ppo"
    learning_rate: float = 3e-5
    max_steps: int | None = None
    save_every: int = 5
    wandb_project: str | None = None
    wandb_name: str | None = None


def build_config(config: HarborTinkerConfig, store_url: str, run_id: str) -> train.Config:
    return train.Config(
        model_name=config.model_name,
        renderer_name=config.renderer_name,
        base_url=config.base_url,
        recipe_name="harbor-tinker",
        log_path=config.log_path,
        dataset_builder=HarborDatasetBuilder(
            tasks_dir=config.tasks_dir,
            groups_per_batch=config.groups_per_batch,
            group_size=config.group_size,
            epochs=config.epochs,
        ),
        rollout_error_tolerance=HarborRolloutStrategy(
            store_url=store_url,
            run_id=run_id,
            renderer_name=config.renderer_name,
            trial_template=Path(config.trial_config).read_text(),
            reward_key=config.reward_key,
            advertised_host=config.advertised_host,
            listen_host=config.listen_host,
            max_parallel_trials_per_group=config.max_parallel_trials_per_group,
        ),
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        loss_fn=config.loss_fn,
        max_steps=config.max_steps,
        save_every=config.save_every,
        ttl_seconds=None,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
    )


def main(config: HarborTinkerConfig):
    run_id = uuid4().hex
    with capture_to_store(run_id, data_dir=Path(config.log_path).expanduser() / "capture") as captured:
        asyncio.run(train.main(build_config(config, captured.base_url, run_id)))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
