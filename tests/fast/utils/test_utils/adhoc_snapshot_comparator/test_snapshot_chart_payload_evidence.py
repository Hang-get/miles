import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_chart_payload_evidence import _explain_pair


@pytest.mark.parametrize(
    "value,verdict", [(False, "historical-replay-environment-difference"), (0, "requires-specific-witness")]
)
def test_router_replay_evidence_requires_the_exact_boolean(value: bool | int, verdict: str) -> None:
    """Numeric zero cannot use a boolean replay witness to explain a missing router input."""
    evidence = _explain_pair(
        pair={
            "path": ["router_args", "json_log"],
            "before": {"present": True, "value": value},
            "after": {"present": False, "value": None},
        },
        original={"parsed": {"router_json_log": False}},
    )
    assert evidence["verdict"] == verdict


def test_payload_dependency_change_preserves_its_exact_version_boundary() -> None:
    """Current payload defaults are explained against the supplied historical observation."""
    pair = {
        "path": ["sglang", "base_args", "file_storage_path"],
        "before": {"present": True, "value": "sglang_storage"},
        "after": {"present": True, "value": None},
    }
    assert _explain_pair(pair=pair, original={})["verdict"] == "pinned-dependency-contract-change"
    pair["after"]["value"] = "unexpected"
    assert _explain_pair(pair=pair, original={})["verdict"] == "requires-specific-witness"


@pytest.mark.parametrize("enabled", [False, True])
def test_payload_critic_default_requires_actual_disabled_consumer(enabled: bool) -> None:
    """A chart payload needs its own critic predicate rather than a runtime case's witness."""
    evidence = _explain_pair(
        pair={
            "path": ["critic_load"],
            "before": {"present": True, "value": "/checkpoint"},
            "after": {"present": True, "value": None},
        },
        original={},
        actual={"train_backend": "megatron", "use_critic": enabled, "critic_load": None},
    )
    assert evidence["verdict"] == ("requires-specific-witness" if enabled else "disabled-critic-consumer-verified")
