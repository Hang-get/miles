import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import tinker
from aiohttp import web
from tinker_cookbook.capture import current_scope


@dataclass(frozen=True)
class TokenTurn:
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    stop_reason: str


@dataclass
class Episode:
    closing: bool = False
    _next_call: int = 0
    _pending: int = 0
    _idle: asyncio.Event = field(default_factory=asyncio.Event)
    _turns: dict[int, TokenTurn] = field(default_factory=dict)
    _failure: BaseException | None = None

    async def sample(self, client, prompt, num_samples, sampling_params):
        if self.closing:
            raise web.HTTPGone(text="trajectory is closed")
        call_id = self._next_call
        self._next_call += 1
        self._pending += 1
        self._idle.clear()
        try:
            input_ids = tuple(prompt.to_ints())
            response = await client.sample_async(
                prompt=prompt, num_samples=num_samples, sampling_params=sampling_params
            )
            assert num_samples == len(response.sequences) == 1, "the official chat proxy samples one sequence"
            sequence = response.sequences[0]
            turn = TokenTurn(input_ids, tuple(sequence.tokens), tuple(sequence.logprobs), sequence.stop_reason)
            assert len(turn.output_ids) == len(turn.logprobs), "each sampled token must have a logprob"
            self._turns[call_id] = turn
            return response
        except tinker.BadRequestError:
            # A rejected prompt generated nothing; the agent may compact and retry.
            raise
        except BaseException as error:
            self._failure = error
            raise
        finally:
            self._pending -= 1
            if not self._pending:
                self._idle.set()

    async def close(self):
        self.closing = True
        if self._pending:
            await self._idle.wait()
        if self._failure is not None:
            raise RuntimeError("trajectory has an incomplete sampling call; discard the rollout group") from self._failure

    @property
    def turns(self) -> tuple[TokenTurn, ...]:
        assert self.closing and not self._pending, "close and drain the trajectory before collecting tokens"
        if self._failure is not None:
            raise RuntimeError("cannot train an incomplete trajectory") from self._failure
        return tuple(self._turns[index] for index in sorted(self._turns))


class RecordingSampler:
    def __init__(self, sampling_client):
        self._sampling_client = sampling_client
        self._episodes: dict[int, Episode] = {}

    @asynccontextmanager
    async def episode(self, trajectory_id: int):
        if trajectory_id in self._episodes:
            raise ValueError(f"trajectory {trajectory_id} was already registered")
        episode = Episode()
        self._episodes[trajectory_id] = episode
        try:
            yield episode
        finally:
            await episode.close()

    async def sample_async(self, prompt, num_samples, sampling_params):
        trajectory_id = current_scope().get("traj_idx")
        if trajectory_id not in self._episodes:
            raise web.HTTPNotFound(text="unknown trajectory")
        return await self._episodes[trajectory_id].sample(
            self._sampling_client, prompt, num_samples, sampling_params
        )
