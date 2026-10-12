from types import SimpleNamespace

import pytest
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


@pytest.fixture
def renderer():
    return RoleColonRenderer(Tokenizer())


@pytest.fixture
def sampler_factory():
    return Sampler
