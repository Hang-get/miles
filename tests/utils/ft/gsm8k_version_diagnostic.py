import gc
import hashlib
import json
import logging
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM

from miles.utils.external_utils.command_utils.common import run_shell_command
from miles.utils.hf_utils.config import load_hf_config

logger = logging.getLogger(__name__)


def run_diagnostic(run: Callable[[], None]) -> None:
    disk = Path("/data/miles_ci")
    run_shell_command("df -hT /data/miles_ci")
    usage = shutil.disk_usage(disk)
    assert usage.total >= 10**12 and usage.free >= 400 * 10**9, usage
    root = disk / "diagnostics" / f"gsm8k-versions-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=False)
    os.environ["MILES_TEST_DUMPS_ROOT"] = str(root)
    print(f"LOGPROB_DIAGNOSTIC_ROOT={root}", flush=True)

    try:
        run()
    finally:
        try:
            run_shell_command("ray stop --force")
            _replay_saved_batches(root)
        except Exception:
            logger.exception("Independent logprob replay could not complete")
            raise


def _replay_saved_batches(root: Path) -> None:
    files = [path for path in root.rglob("samples/*.pt") if path.stem.isdigit()]
    assert files, "No rollout samples were saved"
    files.sort(key=lambda path: int(path.stem))
    for sample_path in dict.fromkeys([files[0], files[-1]]):
        _replay_batch(sample_path)


def _replay_batch(sample_path: Path) -> None:
    batch = torch.load(sample_path, weights_only=False, map_location="cpu")
    with sample_path.open("rb") as stream:
        sample_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    samples = batch["samples"]
    rollout_id = batch["rollout_id"]
    weights_root = sample_path.parent.parent / "weights"
    current = weights_root / f"before_rollout_{rollout_id}"
    current_metadata = json.loads((current / "diagnostic-version.json").read_text())
    assert current_metadata["rollout_id"] == rollout_id
    snapshots = {}
    for path in sorted(weights_root.glob("*/diagnostic-version.json")):
        metadata = json.loads(path.read_text())
        version = metadata["weight_version"]
        if version in snapshots:
            existing = snapshots[version]
            assert _weight_digests(existing) == _weight_digests(
                path.parent
            ), f"Version {version} identifies different weights: {existing}, {path.parent}"
        else:
            snapshots[version] = path.parent

    by_version: dict[int, list[dict[str, Any]]] = {}
    excluded = []
    for sample in samples:
        versions = {
            int(span["version"])
            for call in sample["weight_versions"]
            for span in call["spans"] + call["prefill_spans"]
        }
        if len(versions) != 1:
            excluded.append(dict(index=sample["index"], versions=sorted(versions)))
            continue
        version = versions.pop()
        by_version.setdefault(version, []).append(sample)

    print(json.dumps(dict(rollout_id=rollout_id, current=current_metadata, excluded=excluded)), flush=True)
    for version, group in sorted(by_version.items()):
        assert version in snapshots, f"Missing actual weight version {version}"
        for name, checkpoint in [("same_version", snapshots[version]), ("current_version", current)]:
            results = _recompute(checkpoint=checkpoint, samples=group)
            payload = dict(
                rollout_id=rollout_id,
                source_version=version,
                checkpoint=str(checkpoint),
                checkpoint_sha256=_weight_digests(checkpoint),
                sample_sha256=sample_sha256,
                total_samples=len(samples),
                excluded_samples=excluded,
                mode=name,
                count=len(results),
                mean_saved=sum(item["saved"] for item in results) / len(results),
                mean_recomputed=sum(item["recomputed"] for item in results) / len(results),
                mean_token_abs_diff=sum(item["token_abs_diff"] for item in results) / len(results),
                samples=results,
            )
            out = sample_path.parent.parent / f"replay-{rollout_id}-{version}-{name}.json"
            out.write_text(json.dumps(payload))
            print("LOGPROB_REPLAY " + json.dumps(payload), flush=True)


@torch.inference_mode()
def _recompute(*, checkpoint: Path, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assert (checkpoint / ".complete").is_file(), f"Incomplete export: {checkpoint}"
    model, loading_info = Qwen2ForCausalLM.from_pretrained(
        checkpoint,
        config=load_hf_config(str(checkpoint)),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        output_loading_info=True,
    )
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
        assert not loading_info[key], (checkpoint, key, loading_info[key])
    model = model.to("cuda").eval()
    results = []
    for sample in samples:
        tokens = torch.tensor(sample["tokens"], device="cuda", dtype=torch.long).unsqueeze(0)
        response_length = sample["response_length"]
        prompt_length = tokens.shape[1] - response_length
        logits = model(tokens, use_cache=False).logits[0, prompt_length - 1 : -1].float()
        labels = tokens[0, prompt_length:]
        recomputed = -F.cross_entropy(logits, labels, reduction="none")
        saved = torch.tensor(sample["rollout_log_probs"], device="cuda")
        mask = torch.tensor(sample["loss_mask"] or [1] * response_length, device="cuda", dtype=torch.float32)
        assert recomputed.shape == saved.shape == mask.shape
        assert mask.sum() > 0
        assert torch.isfinite(recomputed).all() and torch.isfinite(saved).all()
        results.append(
            dict(
                index=sample["index"],
                saved=((saved * mask).sum() / mask.sum()).item(),
                recomputed=((recomputed * mask).sum() / mask.sum()).item(),
                token_abs_diff=(((recomputed - saved).abs() * mask).sum() / mask.sum()).item(),
            )
        )
        del logits, recomputed, tokens
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def _weight_digests(checkpoint: Path) -> dict[str, str]:
    paths = sorted(checkpoint.glob("*.safetensors"))
    assert paths, f"No exported weights in {checkpoint}"
    result = {}
    for path in paths:
        with path.open("rb") as stream:
            result[path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result
