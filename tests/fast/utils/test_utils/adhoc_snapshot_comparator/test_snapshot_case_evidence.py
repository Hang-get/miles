import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_case_evidence import _case_evidence
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference, SnapshotValue


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
    from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference, SnapshotValue

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
