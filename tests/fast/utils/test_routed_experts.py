import struct
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
import zstandard

from miles.rollout.generate_utils.generate_endpoint_utils import _merge_routed_experts, request_routed_experts_patch
from miles.utils.routed_experts import decode_routed_experts, routed_experts_dtype
from miles.utils.types import Sample

U8 = np.dtype(np.uint8)


def _routing(num_tokens, num_layers=3, topk=4, num_experts=128, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.argsort(rng.random((num_tokens, num_layers, num_experts)), axis=-1)[..., :topk]
    return np.sort(rows, axis=-1).astype(np.int32)


def _base64_int32(rows):
    return {"routed_experts": pybase64.b64encode(rows.astype(np.int32).tobytes()).decode("ascii")}


def _zstd(rows, dtype=np.uint8):
    """Mirror of sglang.srt.state_capturer.routed_experts_codec (zstd encoding)."""
    layer_major = np.ascontiguousarray(rows.transpose(1, 0, 2).astype(dtype))
    header = struct.pack("<BIHH", np.dtype(dtype).itemsize, *rows.shape)
    frame = zstandard.ZstdCompressor(level=1).compress(layer_major.tobytes())
    return {"routed_experts": pybase64.b64encode(header + frame).decode("ascii"), "routed_experts_encoding": "zstd"}


@pytest.mark.parametrize("encode", [_base64_int32, _zstd])
def test_decode_round_trips_and_narrows(encode):
    rows = _routing(9)
    out = decode_routed_experts(encode(rows), 9, 3, U8)
    assert out.dtype == U8 and out.flags.c_contiguous
    np.testing.assert_array_equal(out, rows)


@pytest.mark.parametrize("encode", [_base64_int32, _zstd])
def test_decode_drops_stop_edge_tail_row(encode):
    rows = _routing(6)
    out = decode_routed_experts(encode(rows), 5, 3, U8, allow_extra_tail_row=True)
    np.testing.assert_array_equal(out, rows[:5])
    with pytest.raises(ValueError, match="do not match"):
        decode_routed_experts(encode(rows), 5, 3, U8)


def test_decode_keeps_int32_and_widens_uint8_payloads():
    rows = _routing(4)
    np.testing.assert_array_equal(decode_routed_experts(_zstd(rows), 4, 3, np.dtype(np.int32)), rows)
    assert decode_routed_experts(_zstd(rows), 4, 3, np.dtype(np.uint16)).dtype == np.uint16


def test_decode_rejects_ids_that_do_not_fit():
    rows = _routing(4, num_experts=384)
    rows[0, 0, -1] = 383
    with pytest.raises(ValueError, match="do not fit uint8"):
        decode_routed_experts(_base64_int32(rows), 4, 3, U8)


def test_decode_missing_payload_and_empty():
    assert decode_routed_experts({}, 4, 3, U8) is None
    assert decode_routed_experts({"routed_experts": ""}, 0, 3, U8).shape == (0, 3, 0)


def test_decode_rejects_all_zero_payload():
    with pytest.raises(AssertionError, match="all zeros"):
        decode_routed_experts(_base64_int32(np.zeros((2, 3, 4), np.int32)), 2, 3, U8)


def test_decode_rejects_unknown_encoding():
    with pytest.raises(ValueError, match="unknown routed_experts_encoding"):
        decode_routed_experts({"routed_experts": "", "routed_experts_encoding": "lz4"}, 0, 3, U8)


@pytest.mark.parametrize(
    "num_experts,dtype", [(None, np.int32), (128, np.uint8), (256, np.uint8), (384, np.uint16), (70000, np.int32)]
)
def test_routed_experts_dtype(num_experts, dtype):
    assert routed_experts_dtype(SimpleNamespace(num_experts=num_experts)) == dtype


def _args(pause_generation_mode="in_place"):
    return SimpleNamespace(num_layers=3, num_experts=128, pause_generation_mode=pause_generation_mode)


def test_additional_r3_requests_and_appends_only_new_rows():
    rows = _routing(10)
    sample = Sample(tokens=list(range(7)), rollout_routed_experts=rows[:6].astype(np.uint8))
    payload = {"input_ids": list(range(8)), "return_routed_experts": True}
    request_routed_experts_patch(_args(), payload, sample)
    assert payload["routed_experts_start_len"] == 6

    sample.tokens = list(range(11))  # input (8) + 3 generated tokens -> 10 rows
    merged = _merge_routed_experts(_args(), sample, payload, {"meta_info": _zstd(rows[6:])})
    assert merged.dtype == U8
    np.testing.assert_array_equal(merged, rows)


@pytest.mark.parametrize("pause_generation_mode", ["retract"])
def test_additional_r3_off_when_prefix_may_be_recomputed(pause_generation_mode):
    sample = Sample(tokens=list(range(7)), rollout_routed_experts=_routing(6).astype(np.uint8))
    payload = {"input_ids": list(range(8)), "return_routed_experts": True}
    request_routed_experts_patch(_args(pause_generation_mode), payload, sample)
    assert "routed_experts_start_len" not in payload


def test_additional_r3_off_on_first_turn_and_without_replay():
    payload = {"input_ids": [1, 2], "return_routed_experts": True}
    request_routed_experts_patch(_args(), payload, Sample(tokens=[1, 2]))
    assert "routed_experts_start_len" not in payload
    payload = {"input_ids": [1, 2], "return_routed_experts": False}
    request_routed_experts_patch(_args(), payload, Sample(tokens=[1, 2], rollout_routed_experts=_routing(1)))
    assert "routed_experts_start_len" not in payload
