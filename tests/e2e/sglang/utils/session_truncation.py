"""Drive one session through turns cut by ``finish_reason='length'`` against a real engine.

Each cut is placed by a probe: the turn is generated in full, the position of a
reasoning, tool-call or answer token is read from it, and the same request is
re-sent with ``max_tokens`` ending there. The engine must decode deterministically
(``--enable-deterministic-inference``, radix cache off, temperature 0), so the
re-sent turn repeats the probe's prefix. Re-sending rolls back the probe in v1 and
branches past it in v2; the picker drops the probe leaf.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from itertools import pairwise

import httpx
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.rollout.session.server import SessionServer
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.http_utils import find_available_port
from miles.utils.processing_utils import load_tokenizer
from miles.utils.test_utils.session_verify_agent import INITIAL_SYSTEM_PROMPT, INITIAL_USER_PROMPT, TOOLS
from miles.utils.test_utils.uvicorn_thread_server import UvicornThreadServer
from miles.utils.types import Sample

_TITO_MODEL = "qwen3"
_PROBE_MAX_TOKENS = 2048
_HARD_MISMATCH_TYPES = frozenset({"special_token_count", "special_token_type", "non_assistant_text"})
_PICKER_PATH = "miles.rollout.session.v2.picker_hub.drop_same_prompt_retries"
_POSTPROCESSOR_PATH = "miles.rollout.session.v2.postprocessor_hub.default_postprocess"

# (follow-up message, where to cut the reply); the first turn has no follow-up.
# The closer text tells which structures a cut left open.
_SCRIPT = (
    (None, "tool_call"),
    ("Your get_weather call was cut off. Call it again for Beijing.", "reasoning"),
    ("Without calling any tool, describe a sunny day in two sentences.", "answer"),
    ("Thanks.", None),
)


def run_truncated_session(*, backend_url: str, model_path: str, version: str) -> None:
    """Cut a reply in a tool call, in reasoning and in the answer, continue each, and check the result."""
    tokenizer = load_tokenizer(model_path, trust_remote_code=True)
    tito = get_tito_tokenizer(tokenizer, tokenizer_type=_TITO_MODEL)
    marker_ids = {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in (("think_end", "</think>"), ("tool_start", "<tool_call>"), ("tool_end", "</tool_call>"))
    }
    with (
        _serve_session(backend_url=backend_url, model_path=model_path, version=version) as url,
        httpx.Client(timeout=600.0) as client,
    ):
        session_url = f"{url}/sessions/" + _post(client, f"{url}/sessions", {"temperature": 0.0})["session_id"]
        messages = [
            {"role": "system", "content": INITIAL_SYSTEM_PROMPT},
            {"role": "user", "content": INITIAL_USER_PROMPT},
        ]
        for follow_up, cut in _SCRIPT:
            if follow_up is not None:
                messages.append({"role": "user", "content": follow_up})
            if cut is None:
                reply = _chat(client, session_url, messages, max_tokens=_PROBE_MAX_TOKENS)
            else:
                reply = _cut_reply(client, session_url, messages, cut=cut, marker_ids=marker_ids)
                # The cut turn is now the session's last turn, compared closed.
                _assert_no_hard_mismatch(client.get(session_url).json(), context=f"{version} after the {cut} cut")
            messages.append(reply["choices"][0]["message"])

        snapshot = client.get(session_url).json()
        response = client.post(f"{session_url}/samples", json={})
        assert response.status_code == 200, response.text

    (sample,) = decode_samples_and_merge_input_sample(response.content, Sample()).samples
    kinds = _assert_cut_turns_closed(snapshot["records"], tito=tito, tokenizer=tokenizer, version=version)
    assert kinds == [cut for _, cut in _SCRIPT if cut is not None], f"{version} cut kinds: {kinds}"
    _assert_no_hard_mismatch(snapshot, context=f"{version} final")
    _assert_closers_masked(sample, snapshot["records"], version=version)


@contextmanager
def _serve_session(*, backend_url: str, model_path: str, version: str):
    port = find_available_port(31000)
    config = make_session_server_config(
        port=port,
        instance_id=uuid.uuid4().hex,
        backend_url=backend_url,
        timeout=600,
        hf_checkpoint=model_path,
        tito_model=_TITO_MODEL,
        use_session_server=version,
        pause_generation_mode="retract",
        session_sample_picker_path=_PICKER_PATH,
        session_sample_postprocessor_path=_POSTPROCESSOR_PATH,
    )
    server = UvicornThreadServer(SessionServer(config).app, host="127.0.0.1", port=port)
    server.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.stop()


def _post(client: httpx.Client, url: str, body: dict) -> dict:
    response = client.post(url, json=body)
    assert response.status_code == 200, f"{url}: {response.status_code} {response.text}"
    return response.json()


def _chat(client: httpx.Client, session_url: str, messages: list[dict], *, max_tokens: int) -> dict:
    body = {"messages": messages, "tools": TOOLS, "max_tokens": max_tokens}
    return _post(client, f"{session_url}/v1/chat/completions", body)


def _cut_reply(client: httpx.Client, session_url: str, messages: list[dict], *, cut: str, marker_ids: dict) -> dict:
    probe = _chat(client, session_url, messages, max_tokens=_PROBE_MAX_TOKENS)
    assert probe["choices"][0]["finish_reason"] != "length", f"probe for the {cut} cut hit {_PROBE_MAX_TOKENS} tokens"
    probe_ids = _completion_ids(client.get(session_url).json()["records"][-1])
    reply = _chat(client, session_url, messages, max_tokens=_cut_length(probe_ids, cut=cut, marker_ids=marker_ids))
    assert reply["choices"][0]["finish_reason"] == "length", f"the {cut} cut was not truncated: {reply}"
    return reply


def _cut_length(probe_ids: list[int], *, cut: str, marker_ids: dict) -> int:
    """How many of the probe's tokens to keep so the re-sent turn stops inside *cut*."""
    text = f"probe={probe_ids}"
    assert marker_ids["think_end"] in probe_ids, f"probe has no reasoning: {text}"
    think_end = probe_ids.index(marker_ids["think_end"])
    match cut:
        case "reasoning":
            assert think_end >= 4, f"reasoning too short to cut: {text}"
            return think_end // 2
        case "tool_call":
            assert marker_ids["tool_start"] in probe_ids, f"probe made no tool call: {text}"
            start = probe_ids.index(marker_ids["tool_start"])
            assert probe_ids.index(marker_ids["tool_end"]) > start + 4, f"tool call too short to cut: {text}"
            return start + 4
        case "answer":
            assert marker_ids["tool_start"] not in probe_ids, f"probe made a tool call instead of answering: {text}"
            answer_start, answer_end = think_end + 1, len(probe_ids) - 1
            assert answer_end - answer_start >= 4, f"answer too short to cut: {text}"
            return (answer_start + answer_end) // 2
    raise ValueError(f"unknown cut {cut!r}")


def _completion_ids(record: dict) -> list[int]:
    return [entry[1] for entry in record["response"]["choices"][0]["meta_info"]["output_token_logprobs"]]


def _assert_cut_turns_closed(records: list[dict], *, tito, tokenizer, version: str) -> list[str]:
    """Check each cut turn's continuation and return the kind of each cut, in order.

    The next prompt must hold the closer recomputed from the stored tokens right
    after the cut, and the closed turn must have no open or misnested structure.
    """
    kinds = []
    for record, next_record in pairwise(records):
        if record["response"]["choices"][0]["finish_reason"] != "length":
            continue
        prompt_ids = record["request"]["input_ids"]
        cut_turn = prompt_ids + _completion_ids(record)
        closer = tito.close_truncated_turn(cut_turn, len(prompt_ids))
        next_prompt = next_record["request"]["input_ids"]
        assert next_prompt[: len(cut_turn)] == cut_turn, f"{version}: the cut turn's tokens changed"
        assert next_prompt[len(cut_turn) : len(cut_turn) + len(closer)] == closer, (
            f"{version} truncation_closer: expected {tokenizer.decode(closer)!r}, got "
            f"{tokenizer.decode(next_prompt[len(cut_turn) : len(cut_turn) + len(closer)])!r}"
        )
        problems = tito.check_turn_structure(next_prompt, (len(prompt_ids), len(cut_turn) + len(closer)))
        assert problems == [], f"{version} unclosed_structure: {problems}"
        kinds.append(_cut_kind(tokenizer.decode(closer)))
    return kinds


def _cut_kind(closer_text: str) -> str:
    if "</tool_call>" in closer_text:
        return "tool_call"
    return "reasoning" if "</think>" in closer_text else "answer"


def _assert_no_hard_mismatch(snapshot: dict, *, context: str) -> None:
    mismatches = snapshot["metadata"].get("tito_session_mismatch")
    assert mismatches is not None, f"{context}: mismatch check failed"
    hard = [m for m in mismatches if m["type"] in _HARD_MISMATCH_TYPES]
    assert hard == [], f"{context}: {json.dumps(hard)[:2000]}"


def _assert_closers_masked(sample: Sample, records: list[dict], *, version: str) -> None:
    """Sampled tokens of every turn train; the tokens between turns, closers included, do not."""
    response_start = len(sample.tokens) - sample.response_length
    trained = {response_start + i for i, bit in enumerate(sample.loss_mask) if bit}
    sampled = set()
    for record in records:
        start = len(record["request"]["input_ids"])
        sampled.update(range(start, start + len(_completion_ids(record))))
    assert sample.tokens == records[-1]["request"]["input_ids"] + _completion_ids(records[-1]), f"{version}: tokens"
    assert trained == {p for p in sampled if p >= response_start}, f"{version}: loss mask"
