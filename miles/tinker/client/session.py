import asyncio
import time
import uuid
from dataclasses import dataclass, field

from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.renderers import Renderer
from tinker_cookbook.rl.types import StopReason

import tinker
from miles.tinker.client.rendering import (
    ChatRequest,
    ChatRequestError,
    parse_completion,
    render_prompt,
    sampling_params,
)
from miles.tinker.core.token_trace import TokenTrace


@dataclass(frozen=True)
class Refusal:
    """A session limit that refused a request, with the token that refusal carried to the harness."""

    limit: StopReason
    token: str


class SessionLimit(ChatRequestError):
    def __init__(self, refusal: Refusal, reason: str):
        super().__init__(reason)
        self.refusal = refusal


@dataclass
class ChatSession:
    policy: TinkerTokenCompleter
    renderer: Renderer
    max_datum_tokens: int
    max_turns: int = 512
    trace: TokenTrace = field(default_factory=TokenTrace)
    closed: bool = False
    refusal: Refusal | None = None  # the latest limit refusal; a failure that quotes its token is attributed to it
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _refuse(self, limit: StopReason, reason: str) -> SessionLimit:
        self.refusal = Refusal(limit, uuid.uuid4().hex)
        return SessionLimit(self.refusal, reason)

    async def complete(self, request: ChatRequest) -> dict:
        async with self.lock:
            if self.closed:
                raise ChatRequestError("session is closed")
            if len(self.trace.turns) >= self.max_turns:
                raise self._refuse(StopReason.MAX_TURNS, "session turn limit reached")
            # HF fast tokenizers are mutated on every encode, so every tokenizer call stays on this thread.
            input_ids = render_prompt(self.renderer, request)
            params = sampling_params(self.renderer, request)
            params["max_tokens"] = min(params.get("max_tokens", self.policy.max_tokens), self.policy.max_tokens)
            params.setdefault("temperature", self.policy.temperature)
            if self.policy.context_window is not None:
                params["max_tokens"] = min(params["max_tokens"], self.policy.context_window - len(input_ids))
            if params["max_tokens"] <= 0 or len(input_ids) + params["max_tokens"] - 1 > self.max_datum_tokens:
                raise self._refuse(StopReason.CONTEXT_OVERFLOW, "prompt plus completion exceeds the per-datum token budget")
            response = await self.policy.sampling_client.sample_async(
                prompt=tinker.ModelInput.from_ints(input_ids),
                num_samples=1,
                sampling_params=tinker.SamplingParams(**params),
            )
            sequence = response.sequences[0]
            # Parsing is a display operation; its failure must not erase a completed sample.
            turn = self.trace.record(
                f"chatcmpl-{uuid.uuid4().hex}", input_ids, sequence.tokens, sequence.logprobs, sequence.stop_reason
            )
            message = parse_completion(self.renderer, turn)
            return {
                "id": turn.id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model or "tinker",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": (
                            "tool_calls"
                            if turn.stop_reason == "stop" and message.get("tool_calls")
                            else turn.stop_reason
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": len(turn.input_ids),
                    "completion_tokens": len(turn.output_ids),
                    "total_tokens": len(turn.input_ids) + len(turn.output_ids),
                },
            }
