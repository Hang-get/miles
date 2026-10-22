import json

from pydantic import JsonValue

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference

_OLD_PIN = "f7bd7caef8231ea76406f86dedcd9b2b2579cf1a"
_CURRENT_PIN = "203f4bbc70e54b8e5e8282262cdb88752068a99e"
_CHANGES = {
    "attn_dp_size": (
        False,
        None,
        True,
        1,
        "parallel.py",
        "The new explicit attention-DP declaration starts at one; enable_dp_attention is resolved from the attention-DP topology.",
    ),
    "cuda_graph_tc_compiler": (
        True,
        None,
        False,
        None,
        "exec_.py",
        "The obsolete compiler selector is absent from the current argument schema; this is a real CLI surface removal.",
    ),
    "disaggregation_decode_allocation_policy": (
        False,
        None,
        True,
        "early",
        "disagg.py",
        "The allocation policy is now explicit; serving_hook and pd_disaggregation_hook distinguish early allocation from prefill_complete.",
    ),
    "disaggregation_decode_host_receive_threshold": (
        True,
        0.0,
        True,
        1.0,
        "disagg.py",
        "The disabled sentinel changes from zero to one together with the guard: old host receive required threshold > 0, current host receive requires threshold < 1. Both recorded defaults disable it; explicit nondefault values have a changed contract.",
    ),
    "dsa_triton_union": (
        False,
        None,
        True,
        0,
        "exec_.py",
        "The new DSA union selector is explicitly zero; other values select the new implementation and are not covered by this value-specific rationale.",
    ),
    "enable_dsa_fused_indexer": (
        False,
        None,
        True,
        None,
        "exec_.py",
        "The new indexer selector is unset, leaving selection to backend/model resolution. This source declaration does not prove final engine dispatch equivalence.",
    ),
    "enable_kv_cache_sharding": (
        False,
        None,
        True,
        False,
        "parallel.py",
        "kv_shard_hook returns before enabling sharding when this new switch is false.",
    ),
    "enable_watermark": (
        False,
        None,
        True,
        False,
        "exec_.py",
        "validation_hook returns without watermark validation when this new switch is false.",
    ),
    "file_storage_path": (
        True,
        "sglang_storage",
        True,
        None,
        "serving.py",
        "The old file backend did not consume this argument; the new unified_radix_cache forwards only a truthy override. With the recorded None, HiCacheFile still uses its environment override or /tmp/hicache fallback. Explicit nonempty paths now affect storage.",
    ),
    "qsa_indexer_dtype": (
        False,
        None,
        True,
        "auto",
        "model.py",
        "The new QSA dtype selector uses auto resolution; this schema evidence does not certify model-specific final dtype selection.",
    ),
    "speculative_enable_w4a4_mxfp4_megamoe": (
        False,
        None,
        True,
        None,
        "spec.py",
        "The new speculative quantization selector is unset; the value is a dependency declaration, not a mechanical field move.",
    ),
    "speculative_use_block_verification": (
        False,
        None,
        True,
        False,
        "spec.py",
        "speculative_hook and dflash_worker_v2 use the false switch to retain the non-block verification path.",
    ),
    "watermark_config": (
        False,
        None,
        True,
        None,
        "exec_.py",
        "The new optional watermark config is absent; validation_hook rejects a configured value without the enable switch, so this rule accepts only None.",
    ),
}


def _dependency_evidence(*, difference: SnapshotDifference) -> dict[str, JsonValue] | None:
    field = difference.path[-1]
    if field.startswith("sglang_"):
        field = field.removeprefix("sglang_")
    elif len(difference.path) < 3 or difference.path[-3:-1] != ["sglang", "base_args"]:
        return None
    if field not in _CHANGES:
        return None
    before_present, before_value, after_present, after_value, source, reason = _CHANGES[field]
    observed = [difference.before.present, difference.before.value, difference.after.present, difference.after.value]
    if json.dumps(observed) != json.dumps([before_present, before_value, after_present, after_value]):
        return None
    # These are pinned dependency changes, not permissions to mutate the before
    # snapshot. Every exact typed pair stays in the strict failure report.
    return {
        "verdict": "pinned-dependency-contract-change",
        "field": field,
        "before_dependency": _OLD_PIN,
        "current_dependency": _CURRENT_PIN,
        "source": "python/sglang/srt/arg_groups/fields/" + source,
        "explanation": reason,
        "capture_boundary": "Historical capture pins differ by batch; the current pin is the source checked here, not an asserted capture provenance.",
    }
