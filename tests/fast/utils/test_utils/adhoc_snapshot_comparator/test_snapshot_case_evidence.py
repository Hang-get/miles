import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_case_evidence import _case_evidence
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotDifference,
    SnapshotValue,
)


@pytest.mark.parametrize("rank,adapter,verified", [(0, None, True), (8, None, False), (0, "/adapter", False)])
def test_lora_null_evidence_requires_both_disabled_consumer_inputs(
    rank: int, adapter: str | None, verified: bool
) -> None:
    """A null declaration is harmless only when the actual consumer is disabled."""
    difference = SnapshotDifference(
        path=["parsed_config/case", "config", "hf_lora_targets"],
        before=SnapshotValue(present=False, value=None),
        after=SnapshotValue(present=True, value=None),
    )
    actual = {"parsed_config/case": {"config": {"lora_rank": rank, "lora_adapter_path": adapter}}}
    assert (_case_evidence(difference=difference, actual=actual) is not None) is verified


def test_new_negative_ft_case_does_not_justify_an_unrelated_error() -> None:
    """Only the precise supported-backend rejection gets the negative-contract rationale."""
    difference = SnapshotDifference(
        path=["parsed_config/fsdp_reject_default_ft.yaml"],
        before=SnapshotValue(present=False, value=None),
        after=SnapshotValue(
            present=True, value={"error": {"message": "unexpected failure", "type": "AssertionError"}}
        ),
    )
    assert _case_evidence(difference=difference, actual={}) is None


def test_critic_default_explanation_requires_a_disabled_actual_consumer() -> None:
    """An unset critic source is safe evidence only for a captured disabled critic."""
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_case_evidence import _case_evidence
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
        SnapshotDifference,
        SnapshotValue,
    )

    path = ["parsed_config/case.yaml", "config", "critic_load"]
    difference = SnapshotDifference(
        path=path, before=SnapshotValue(present=True, value="/actor"), after=SnapshotValue(present=True, value=None)
    )
    args = {"train_backend": "megatron", "use_critic": False, "critic_load": None}
    assert (
        _case_evidence(difference=difference, actual={path[0]: {"config": args}})["verdict"]
        == "disabled-critic-consumer-verified"
    )
    args["use_critic"] = True
    assert _case_evidence(difference=difference, actual={path[0]: {"config": args}}) is None


@pytest.mark.parametrize(
    "backend,stage,before_present,value,verified",
    [
        ("fsdp", "checkpoint_load-0000", False, None, True),
        ("megatron", "checkpoint_load-0000", False, None, False),
        ("torchtitan", "checkpoint_load-0000", False, None, False),
        (None, "checkpoint_load-0000", False, None, False),
        ("fsdp", "process_config-0000", False, None, False),
        ("fsdp", "checkpoint_load-0000", True, None, False),
        ("fsdp", "checkpoint_load-0000", False, 0, False),
    ],
)
def test_fsdp_checkpoint_default_requires_exact_backend_stage_and_value(
    backend: str | None, stage: str, before_present: bool, value: int | None, verified: bool
) -> None:
    """FSDP tracker fallback explains only an absent-to-null step at its load boundary."""
    path = ["runtime_config/case.yaml", "actor", "stages", stage, "args", "backend", "ckpt_step"]
    difference = SnapshotDifference(
        path=path,
        before=SnapshotValue(present=before_present, value=None),
        after=SnapshotValue(present=True, value=value),
    )
    actual = {path[0]: {"actor": {"stages": {stage: {"args": {"backend": {"backend_name": backend}}}}}}}
    evidence = _case_evidence(difference=difference, actual=actual)
    assert (evidence is not None) is verified
    if verified:
        assert evidence["verdict"] == "consumer-default-preserved"


def test_checkpoint_catalog_does_not_infer_megatron_from_field_name() -> None:
    """A checkpoint field alone cannot establish the request/apply consumer contract."""
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_evidence_catalog import _describe_difference

    difference = SnapshotDifference(
        path=["runtime_config/case.yaml", "actor", "stages", "checkpoint_load-0000", "args", "backend", "ckpt_step"],
        before=SnapshotValue(present=False, value=None),
        after=SnapshotValue(present=True, value=None),
    )
    explanation = _describe_difference(difference=difference, category="lifecycle/checkpoint-declaration")
    assert explanation["verdict"] == "checkpoint-consumer-evidence-required"
