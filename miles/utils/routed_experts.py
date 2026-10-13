"""Decode SGLang routed-experts payloads for rollout routing replay (R3).

``meta_info["routed_experts_encoding"]`` names the wire format (absent: ``base64_int32``,
the int32 ``[tokens, layers, topk]`` buffer); ``zstd`` is a ``<BIHH`` header (dtype
itemsize, tokens, layers, topk) then one zstd frame of the ``[layers, tokens, topk]``
array, as written by ``sglang.srt.state_capturer.routed_experts_codec``.
"""

import functools
import struct
from argparse import Namespace

import numpy as np
import pybase64
import zstandard

BASE64_INT32 = "base64_int32"
ZSTD = "zstd"

# Every dtype `routed_experts_dtype` can pick; wire codecs accept exactly these.
ROUTED_EXPERTS_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.int32))

_ZSTD_HEADER = struct.Struct("<BIHH")
_ITEMSIZE_TO_DTYPE = {1: np.dtype(np.uint8), 2: np.dtype(np.uint16), 4: np.dtype(np.int32)}
_ROW_DTYPES = {1: np.dtype(np.uint8), 2: np.dtype(np.uint16), 4: np.dtype(np.uint32), 8: np.dtype(np.uint64)}
_TRANSPOSE_BLOCK = 1024


def routed_experts_dtype(args: Namespace) -> np.dtype:
    """Narrowest dtype holding every expert id of the model; int32 when the count is unknown."""
    num_experts = getattr(args, "num_experts", None) or _hf_num_routed_experts(getattr(args, "hf_checkpoint", None))
    if num_experts is None:
        return np.dtype(np.int32)
    if num_experts <= np.iinfo(np.uint8).max + 1:
        return np.dtype(np.uint8)
    if num_experts <= np.iinfo(np.uint16).max + 1:
        return np.dtype(np.uint16)
    return np.dtype(np.int32)


@functools.cache
def _hf_num_routed_experts(hf_checkpoint: str | None) -> int | None:
    if hf_checkpoint is None:
        return None
    from miles.utils.hf_utils.config import load_hf_config

    config = load_hf_config(hf_checkpoint)
    config = getattr(config, "text_config", None) or config
    for key in ("n_routed_experts", "num_experts", "num_local_experts"):
        if value := getattr(config, key, None):
            return int(value)
    return None


def decode_routed_experts(
    meta_info: dict,
    num_tokens: int,
    num_layers: int,
    dtype: np.dtype,
    *,
    allow_extra_tail_row: bool = False,
) -> np.ndarray | None:
    """``[num_tokens, num_layers, topk]`` rows in ``dtype``, or None if the response has none.

    ``allow_extra_tail_row``: SGLang may also forward the final token at a stop edge;
    that row feeds no training position and is dropped.
    """
    payload = meta_info.get("routed_experts")
    if payload is None:
        return None
    encoding = meta_info.get("routed_experts_encoding", BASE64_INT32)
    data = pybase64.b64decode(payload.encode("ascii"))
    if encoding == BASE64_INT32:
        rows = _int32_rows(data, num_tokens, num_layers, allow_extra_tail_row)
    elif encoding == ZSTD:
        rows = _zstd_rows(data)
    else:
        raise ValueError(f"unknown routed_experts_encoding {encoding!r}")
    if allow_extra_tail_row and rows.shape[0] == num_tokens + 1:
        rows = rows[:num_tokens]
    if rows.shape[0] != num_tokens or (num_tokens > 0 and rows.shape[1] != num_layers):
        raise ValueError(f"routed_experts rows {rows.shape} do not match ({num_tokens}, {num_layers}, topk)")
    assert rows.size == 0 or rows.any(), (
        "routed_experts payload is all zeros: the sglang engine did not capture routed experts "
        "(topk-bypassing --moe-runner-backend such as flashinfer_trtllm?)."
    )
    return _as_dtype(rows, dtype)


def _int32_rows(data: bytes, num_tokens: int, num_layers: int, allow_extra_tail_row: bool) -> np.ndarray:
    flat = np.frombuffer(data, dtype=np.int32)
    if flat.size == 0:
        return flat.reshape(0, num_layers, 0)
    for num_rows in (num_tokens, num_tokens + 1) if allow_extra_tail_row else (num_tokens,):
        if num_rows > 0 and flat.size % (num_rows * num_layers) == 0:
            return flat.reshape(num_rows, num_layers, -1)
    raise ValueError(f"routed_experts buffer of {flat.size} ids do not match ({num_tokens}, {num_layers}, topk)")


def _zstd_rows(data: bytes) -> np.ndarray:
    itemsize, num_tokens, num_layers, topk = _ZSTD_HEADER.unpack_from(data)
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(data[_ZSTD_HEADER.size :])
    layer_major = np.frombuffer(raw, dtype=_ITEMSIZE_TO_DTYPE[itemsize]).reshape(num_layers, num_tokens, topk)
    return _swap_leading_axes(layer_major)


def _swap_leading_axes(arr: np.ndarray) -> np.ndarray:
    """C-contiguous ``[a, b, topk]`` -> contiguous ``[b, a, topk]``.

    Each ``topk`` row moves as one element, a block of the longer axis at a time so
    both sides of the copy stay in cache; several times faster than a plain strided copy.
    """
    a, b, topk = arr.shape
    if arr.size == 0:
        return arr.reshape(b, a, topk)
    row_bytes = topk * arr.itemsize
    rows = arr.view(_ROW_DTYPES.get(row_bytes, np.dtype(f"V{row_bytes}"))).reshape(a, b)
    out = np.empty((b, a), dtype=rows.dtype)
    if a >= b:
        for start in range(0, a, _TRANSPOSE_BLOCK):
            out[:, start : start + _TRANSPOSE_BLOCK] = rows[start : start + _TRANSPOSE_BLOCK].T
    else:
        for start in range(0, b, _TRANSPOSE_BLOCK):
            out[start : start + _TRANSPOSE_BLOCK] = rows[:, start : start + _TRANSPOSE_BLOCK].T
    return out.view(arr.dtype).reshape(b, a, topk)


def _as_dtype(rows: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if rows.dtype == dtype:
        return rows
    if not np.can_cast(rows.dtype, dtype, casting="safe") and rows.size:
        lo, hi = int(rows.min()), int(rows.max())
        if lo < np.iinfo(dtype).min or hi > np.iinfo(dtype).max:
            raise ValueError(f"routed expert ids [{lo}, {hi}] do not fit {dtype}")
    return rows.astype(dtype)
