import asyncio
from types import SimpleNamespace

import httpx
import pytest
from aiohttp import web
from tinker_agent.recording import Episode, RecordingSampler
from tinker_cookbook.capture import capture

import tinker


class ControlledSampler:
    def __init__(self):
        self.started = asyncio.Queue()
        self.results = {}

    async def sample_async(self, prompt, num_samples, sampling_params):
        token = prompt.to_ints()[0]
        future = asyncio.get_running_loop().create_future()
        self.results[token] = future
        self.started.put_nowait(token)
        return await future


def response(token, logprobs=None):
    return SimpleNamespace(
        sequences=[
            SimpleNamespace(tokens=[token], logprobs=[-0.5] if logprobs is None else logprobs, stop_reason="stop")
        ]
    )


async def sample(recorder, trajectory_id, token):
    with capture(traj_idx=trajectory_id):
        return await recorder.sample_async(
            tinker.ModelInput.from_ints([token]), 1, tinker.SamplingParams(max_tokens=8)
        )


@pytest.mark.asyncio
async def test_concurrent_calls_are_isolated_ordered_and_drained():
    backend = ControlledSampler()
    recorder = RecordingSampler(backend)
    async with recorder.episode(0) as first, recorder.episode(1) as second:
        tasks = []
        for trajectory_id, token in [(0, 10), (0, 20), (1, 30)]:
            tasks.append(asyncio.create_task(sample(recorder, trajectory_id, token)))
            assert await backend.started.get() == token
        backend.results[20].set_result(response(2))
        await tasks[1]
        closing = asyncio.create_task(first.close())
        await asyncio.sleep(0)
        assert not closing.done()
        with pytest.raises(web.HTTPGone):
            await sample(recorder, 0, 40)
        backend.results[30].set_result(response(3))
        await tasks[2]
        assert not closing.done()
        backend.results[10].set_result(response(1))
        await tasks[0]
        await closing
    assert [turn.output_ids for turn in first.turns] == [(1,), (2,)]
    assert [turn.output_ids for turn in second.turns] == [(3,)]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "transport", "missing-logprobs"])
async def test_incomplete_call_invalidates_trajectory(failure):
    backend = ControlledSampler()
    recorder = RecordingSampler(backend)
    with pytest.raises(RuntimeError, match="incomplete sampling"):
        async with recorder.episode(0) as episode:
            task = asyncio.create_task(sample(recorder, 0, 1))
            await backend.started.get()
            if failure == "cancel":
                task.cancel()
            elif failure == "transport":
                backend.results[1].set_exception(RuntimeError("lost response"))
            else:
                backend.results[1].set_result(response(2, logprobs=[]))
            with pytest.raises((asyncio.CancelledError, RuntimeError, AssertionError)):
                await task
    with pytest.raises(RuntimeError, match="incomplete trajectory"):
        _ = episode.turns


@pytest.mark.asyncio
async def test_rejected_prompt_can_recover_without_a_phantom_turn():
    backend = ControlledSampler()
    recorder = RecordingSampler(backend)
    async with recorder.episode(0) as episode:
        task = asyncio.create_task(sample(recorder, 0, 1))
        await backend.started.get()
        backend.results[1].set_exception(
            tinker.BadRequestError(
                "prompt too long",
                response=httpx.Response(400, request=httpx.Request("POST", "http://test")),
                body=None,
            )
        )
        with pytest.raises(tinker.BadRequestError):
            await task
        task = asyncio.create_task(sample(recorder, 0, 2))
        await backend.started.get()
        generated = response(3)
        backend.results[2].set_result(generated)
        await task
        generated.sequences[0].tokens[0] = 99
        generated.sequences[0].logprobs[0] = 0.0
    assert episode.turns[0].output_ids == (3,)
    assert episode.turns[0].logprobs == (-0.5,)


@pytest.mark.asyncio
async def test_empty_episode_closes_without_waiting():
    episode = Episode()
    await episode.close()
    assert episode.turns == ()
