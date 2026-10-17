import base64
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import zlib
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def install(args: Namespace) -> None:
    from miles.backends.megatron_utils import actor, model
    from miles.utils.audit_utils.event_logger.logger import get_event_logger

    import transformer_engine.pytorch.attention.dot_product_attention.context_parallel as te_context_parallel

    versions = {}
    for package in ("torch", "transformer-engine", "flash-attn", "flash-attn-3"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    source_path = Path(te_context_parallel.__file__)
    logger.info("Numeric diagnostic libraries: %s; CP source %s sha256=%s", versions, source_path, hashlib.sha256(source_path.read_bytes()).hexdigest())

    original_loss = model.loss_function
    loss_index = 0

    def capture_loss(*positional: Any, **keywords: Any) -> Any:
        nonlocal loss_index
        batch, logits = positional[1], positional[3]
        identity = get_event_logger().source.to_name()
        root = Path(args.save_debug_event_data).parent / "numeric-evidence" / identity / "training"
        index = loss_index
        loss_index += 1
        _save(
            root / f"loss-input-{index}.pt",
            dict(
                logits_stats=_logits_stats(logits),
                batch={key: batch[key] for key in ("log_probs", "advantages", "loss_masks", "sample_indices") if key in batch},
            ),
        )
        result = original_loss(*positional, **keywords)
        _save(root / f"loss-output-{index}.pt", result)
        return result

    model.loss_function = capture_loss

    original_forward = actor.forward_only
    call_index = 0

    def capture_forward(f: Callable[..., Any], *positional: Any, **keywords: Any) -> Any:
        nonlocal call_index
        if keywords["rollout_id"] not in (0, 1):
            return original_forward(f, *positional, **keywords)

        identity = get_event_logger().source.to_name()
        root = Path(args.save_debug_event_data).parent / "numeric-evidence" / identity / str(call_index)
        root.mkdir(parents=True, exist_ok=False)
        call_index += 1
        models, iterators = positional[1], positional[2]
        weight_files = {}
        weights_root = root / "weights"
        weights_root.mkdir()
        for chunk_index, module in enumerate(models):
            for tensor_index, (name, value) in enumerate(module.state_dict().items()):
                path = weights_root / f"{chunk_index}-{tensor_index}.pt"
                _save(path, value)
                weight_files[f"{chunk_index}:{name}"] = str(path.relative_to(root))

        _save(
            root / "initial.pt",
            dict(
                keywords=keywords,
                weight_files=weight_files,
                micro_batch_indices=iterators[0].micro_batch_indices,
                rollout_data={
                    k: v
                    for k, v in iterators[0].rollout_data.items()
                    if k
                    in (
                        "tokens",
                        "total_lengths",
                        "response_lengths",
                        "sample_indices",
                        "loss_masks",
                        "rollout_log_probs",
                        "log_probs",
                        "weight_versions",
                    )
                },
            ),
        )
        original_get_batch = model.get_batch
        original_get_packed = model.get_packed_seq_params
        batch_index = 0
        result_index = 0
        packed_index = 0

        def capture_batch(*a: Any, **kw: Any) -> Any:
            nonlocal batch_index
            result = original_get_batch(*a, **kw)
            _save(root / f"batch-{batch_index}.pt", result)
            batch_index += 1
            return result

        def capture_packed(*a: Any, **kw: Any) -> Any:
            nonlocal packed_index
            result = original_get_packed(*a, **kw)
            _save(root / f"packed-{packed_index}.pt", result)
            packed_index += 1
            return result

        def capture_result(*a: Any, **kw: Any) -> Any:
            nonlocal result_index
            logits = a[0]
            logits_stats = _logits_stats(logits)
            result = f(*a, **kw)
            _save(
                root / f"result-{result_index}.pt",
                dict(inputs={k: v for k, v in kw.items() if k != "args"}, logits_stats=logits_stats, result=result),
            )
            result_index += 1
            return result

        model.get_batch = capture_batch
        model.get_packed_seq_params = capture_packed
        try:
            result = original_forward(capture_result, *positional, **keywords)
            _save(root / "aggregated.pt", result)
            _save(root / "counts.pt", dict(batches=batch_index, packed=packed_index, results=result_index))
            return result
        finally:
            model.get_batch = original_get_batch
            model.get_packed_seq_params = original_get_packed

    actor.forward_only = capture_forward


def _logits_stats(logits: torch.Tensor) -> dict[str, Any]:
    return dict(
        shape=list(logits.shape),
        dtype=str(logits.dtype),
        nan=sum(int(torch.isnan(chunk).sum().item()) for chunk in logits.split(2048, dim=-1)),
        inf=sum(int(torch.isinf(chunk).sum().item()) for chunk in logits.split(2048, dim=-1)),
    )


def _save(path: Path, value: Any) -> None:
    frozen = _freeze(value)
    if path.parent.name == "weights" and isinstance(frozen, torch.Tensor):
        value = dict(
            shape=list(frozen.shape),
            dtype=str(frozen.dtype),
            finite=bool(torch.isfinite(frozen).all()),
            sha256=hashlib.sha256(frozen.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
        )
    else:
        value = _json_value(frozen)
    payload = json.dumps(value).encode()
    digest = hashlib.sha256(payload).hexdigest()
    encoded = base64.b64encode(zlib.compress(payload)).decode()
    chunks = [encoded[i : i + 6000] for i in range(0, len(encoded), 6000)]
    for index, chunk in enumerate(chunks):
        logger.info(
            "MILES_NUMERIC_RECORD %s",
            json.dumps(dict(path=str(path), index=index, count=len(chunks), sha256=digest, data=chunk)),
        )


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return dict(shape=list(value.shape), dtype=str(value.dtype), data=value.tolist())
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_value(v) for v in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Namespace):
        return _freeze(vars(value))
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _freeze(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {k: _freeze(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_freeze(v) for v in value]
    return value
