"""Train from actual renderer/SDK samples, including an edited history and a repeated prompt."""

import argparse
import asyncio

import httpx
from tinker_agent.proxy import serve_proxy
from tinker_agent.recording import RecordingSampler
from tinker_agent.trajectory import turns_to_trajectory
from tinker_cookbook import renderers
from tinker_cookbook.rl.data_processing import trajectory_to_data

import tinker


async def execute(base_url, base_model):
    service = tinker.ServiceClient(base_url=base_url, api_key="tml-session-acceptance")
    training = await service.create_lora_training_client_async(base_model=base_model, rank=8)
    saved = await (await training.save_weights_for_sampler_async(name="session-start"))
    sampler = await service.create_sampling_client_async(model_path=saved.path)
    renderer = renderers.get_renderer("qwen3_instruct", sampler.get_tokenizer())
    recorder = RecordingSampler(sampler)
    datums = []
    async with serve_proxy(
        renderer, recorder, model_label=base_model, default_max_tokens=32
    ) as port, httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=300) as http:
        for trajectory_id, advantage in enumerate([-1.0, 1.0]):
            async with recorder.episode(trajectory_id) as episode:
                path = f"/r/traj/{trajectory_id}"
                messages = [{"role": "user", "content": "Reply with the number 1."}]
                for index in range(3):
                    request = {"messages": messages, "max_tokens": 32}
                    response = await http.post(f"{path}/v1/chat/completions", json=request)
                    response.raise_for_status()
                    if index == 0:
                        messages = [
                            {"role": "user", "content": "Reply with the number 2."},
                            response.json()["choices"][0]["message"],
                            {"role": "user", "content": "Reply with the number 3."},
                        ]
            trajectory = turns_to_trajectory(episode.turns, "completed")
            group_datums = trajectory_to_data(trajectory, traj_advantage=advantage)
            assert len(group_datums) == 3
            assert sum(sum(d.loss_fn_inputs["mask"].data) for d in group_datums) == sum(
                len(turn.output_ids) for turn in episode.turns
            )
            datums.extend(group_datums)
    # The cookbook trainer drops its client-side "mask" before forward_backward (rl/train.py _remove_mask).
    sent = [
        tinker.Datum(
            model_input=datum.model_input,
            loss_fn_inputs={key: value for key, value in datum.loss_fn_inputs.items() if key != "mask"},
        )
        for datum in datums
    ]
    backward = await training.forward_backward_async(sent, loss_fn="ppo")
    step = await training.optim_step_async(tinker.types.AdamParams(learning_rate=1e-5))
    await backward
    await step
    await (await training.save_weights_for_sampler_async(name="session-trained"))
    print("session acceptance passed: 6 sampled turns, 6 Datums, PPO backward + optimizer step")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    asyncio.run(execute(args.base_url, args.base_model))
