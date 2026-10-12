import importlib.util
import json
from types import SimpleNamespace

import httpx
import pytest
from aiohttp.test_utils import TestServer
from tinker_agent.proxy import serve_proxy
from tinker_agent.recording import RecordingSampler
from tinker_cookbook.capture.proxy.app import ProxyDeps, make_app
from tinker_cookbook.renderers.role_colon import RoleColonRenderer


class Tokenizer:
    name_or_path = "fake-tokenizer"
    bos_token = None
    eos_token_id = None

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, tokens, **kwargs):
        return "".join(" hello\n\nUser:" if token == 1000 else chr(token) for token in tokens)


class Sampler:
    def __init__(self):
        self.calls = []

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append((prompt.to_ints(), num_samples, sampling_params.model_dump()))
        return SimpleNamespace(sequences=[SimpleNamespace(tokens=[1000], logprobs=[-0.25], stop_reason="stop")])


def normalized(payload):
    if isinstance(payload, dict):
        return {key: normalized(value) for key, value in payload.items() if key not in {"id", "created"}}
    if isinstance(payload, list):
        return [normalized(value) for value in payload]
    return payload


def response_payload(response):
    if response.headers["content-type"].startswith("text/event-stream"):
        return [
            normalized(json.loads(line[6:]))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
    return normalized(response.json())


def test_client_install_has_no_miles_dependency():
    assert importlib.util.find_spec("miles") is None, "run the installed client tests with python -I"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "endpoint,extra",
    [
        ("chat/completions", {}),
        ("chat/completions", {"stream": True}),
        ("chat/completions", {"stop": ["END"]}),
        ("chat/completions", {"stop": [42]}),
        ("chat/completions", {"response_format": {"type": "json_object"}}),
        ("messages", {}),
        ("messages", {"stream": True, "thinking": {"type": "enabled", "budget_tokens": 16}}),
    ],
)
async def test_recording_preserves_official_proxy_behavior(endpoint, extra):
    renderer = RoleColonRenderer(Tokenizer())
    bare, backend = Sampler(), Sampler()
    recorder = RecordingSampler(backend)
    official = TestServer(make_app(ProxyDeps(renderer, bare, "policy", default_max_tokens=32)))
    await official.start_server()
    try:
        async with serve_proxy(renderer, recorder, model_label="policy", default_max_tokens=32) as port:
            async with recorder.episode(0) as episode, httpx.AsyncClient() as http:
                body = {
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 17,
                    "temperature": 0.7,
                    **extra,
                }
                path = f"/r/traj/0/v1/{endpoint}"
                expected = await http.post(str(official.make_url(path)), json=body)
                actual = await http.post(f"http://127.0.0.1:{port}{path}", json=body)
                assert actual.status_code == expected.status_code
                assert response_payload(actual) == response_payload(expected)
                assert backend.calls == bare.calls
            assert len(episode.turns) == len(backend.calls)
            if episode.turns:
                assert episode.turns[0].input_ids == tuple(backend.calls[0][0])
                assert episode.turns[0].output_ids == (1000,)
    finally:
        await official.close()


@pytest.mark.asyncio
async def test_parse_failure_keeps_sample_and_closed_trajectory_cannot_sample(monkeypatch):
    renderer = RoleColonRenderer(Tokenizer())
    backend = Sampler()
    recorder = RecordingSampler(backend)

    def fail_parse(tokens):
        raise ValueError("broken output parser")

    monkeypatch.setattr(renderer, "parse_response", fail_parse)
    async with serve_proxy(renderer, recorder, model_label="policy", default_max_tokens=32) as port:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as http:
            async with recorder.episode(0) as episode:
                body = {"messages": [{"role": "user", "content": "hi"}]}
                response = await http.post("/r/traj/0/v1/chat/completions", json=body)
                assert response.status_code == 500
            assert episode.turns[0].output_ids == (1000,)
            assert (await http.post("/r/traj/0/v1/chat/completions", json=body)).status_code == 410
            assert (await http.post("/r/traj/1/v1/chat/completions", json=body)).status_code == 404
            assert len(backend.calls) == 1
