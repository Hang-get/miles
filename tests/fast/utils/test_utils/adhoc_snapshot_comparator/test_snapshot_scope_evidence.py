import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import (
    SnapshotDifference,
    SnapshotValue,
)
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_scope_evidence import _ScopeEvidence


class TestSnapshotScopeEvidence:
    @pytest.mark.parametrize(
        "observed,verdict", [(False, "same-generation-value-found"), (0, "same-generation-value-not-found")]
    )
    def test_scope_witness_requires_the_same_typed_value(self, observed: bool | int, verdict: str) -> None:
        """Role omission alone cannot prove equality, including the bool/int equality trap."""
        snapshots = {
            "runtime_config/case": {
                "run/actor_cell00000": {"ranks": [0], "stages": {}},
                "run/rollout_executor": {"stages": {"process_config-0000": {"args": {"partial_rollout": observed}}}},
            }
        }
        evidence = _ScopeEvidence(
            actual=snapshots, fields={"all": frozenset({"partial_rollout"}), "trainer": frozenset()}
        )
        difference = SnapshotDifference(
            path=[
                "runtime_config/case",
                "run/actor_cell00000",
                "stages",
                "process_config-0000",
                "args",
                "partial_rollout",
            ],
            before=SnapshotValue(present=True, value=False),
            after=SnapshotValue(present=False, value=None),
        )
        result = evidence.explain(difference=difference)
        assert result is not None and result["verdict"] == verdict
        assert difference.before.present and not difference.after.present
        if verdict == "same-generation-value-found":
            assert result["witness"][1] == "run/rollout_executor"
        else:
            assert result["witness"] is None

    def test_nested_backend_witness_does_not_alone_verify_ownership(self) -> None:
        """A backend value match still requires a separate owner and consumer audit."""
        actual = {
            "runtime_config/case": {
                "run/main": {
                    "stages": {"process_config-0000": {"args": {"raw_megatron": {"base_args": {"hidden_size": 512}}}}}
                }
            }
        }
        evidence = _ScopeEvidence(
            actual=actual,
            fields={"all": frozenset(), "inference": frozenset()},
            backend_fields=frozenset({"hidden_size"}),
        )
        difference = SnapshotDifference(
            path=[
                "runtime_config/case",
                "run/inference_controller",
                "stages",
                "process_config-0000",
                "args",
                "hidden_size",
            ],
            before=SnapshotValue(present=True, value=512),
            after=SnapshotValue(present=False, value=None),
        )
        result = evidence.explain(difference=difference)
        assert result is not None and result["verdict"] == "same-generation-value-found"
        assert result["witness"][-3:] == ["raw_megatron", "base_args", "hidden_size"]

    def test_a_different_run_cannot_supply_the_missing_owner_witness(self) -> None:
        """A matching value in another generation cannot prove this generation's configuration."""
        actual = {
            "runtime_config/case": {
                "run-0001/all/default/main": {"stages": {"process_config-0000": {"args": {"partial_rollout": False}}}}
            }
        }
        evidence = _ScopeEvidence(
            actual=actual, fields={"all": frozenset({"partial_rollout"}), "inference": frozenset()}
        )
        difference = SnapshotDifference(
            path=[
                "runtime_config/case",
                "run-0000/all/default/inference_controller",
                "stages",
                "process_config-0000",
                "args",
                "partial_rollout",
            ],
            before=SnapshotValue(present=True, value=False),
            after=SnapshotValue(present=False, value=None),
        )
        result = evidence.explain(difference=difference)
        assert result is not None and result["verdict"] == "same-generation-value-not-found"

    @pytest.mark.parametrize(
        "backend,value,verdict",
        [
            ("fsdp", None, "same-generation-inactive-backend-value-found"),
            ("torchtitan", None, "same-generation-inactive-backend-value-found"),
            ("megatron", None, "same-generation-value-not-found"),
            ("fsdp", 128, "same-generation-value-not-found"),
        ],
    )
    def test_inactive_shim_requires_a_same_run_backend_owner_and_null(
        self, backend: str, value: int | None, verdict: str
    ) -> None:
        """Only the unused null shim on an observed FSDP/TorchTitan run has the inactive-field proof."""
        evidence = _ScopeEvidence(
            actual={
                "runtime_config/case": {
                    "run/main": {"stages": {"process_config-0000": {"args": {"train_backend": backend}}}}
                }
            },
            fields={"all": frozenset(), "inference": frozenset()},
            backend_fields=frozenset({"cp_comm_type"}),
        )
        difference = SnapshotDifference(
            path=[
                "runtime_config/case",
                "run/inference_controller",
                "stages",
                "process_config-0000",
                "args",
                "cp_comm_type",
            ],
            before=SnapshotValue(present=True, value=value),
            after=SnapshotValue(present=False, value=None),
        )
        result = evidence.explain(difference=difference)
        assert result is not None and result["verdict"] == verdict

    @pytest.mark.parametrize(
        "debug,verdict",
        [(True, "same-generation-rollout-only-value-found"), (False, "same-generation-value-not-found")],
    )
    def test_checkpoint_omission_requires_rollout_only_in_the_same_run(self, debug: bool, verdict: str) -> None:
        """An active training run cannot borrow the rollout-only checkpoint explanation."""
        evidence = _ScopeEvidence(
            actual={
                "runtime_config/case": {
                    "run/main": {"stages": {"process_config-0000": {"args": {"debug_rollout_only": debug}}}}
                }
            },
            fields={"all": frozenset({"load"}), "inference": frozenset()},
        )
        difference = SnapshotDifference(
            path=["runtime_config/case", "run/inference_controller", "stages", "process_config-0000", "args", "load"],
            before=SnapshotValue(present=True, value="/old/checkpoint"),
            after=SnapshotValue(present=False, value=None),
        )
        result = evidence.explain(difference=difference)
        assert result is not None and result["verdict"] == verdict

    @pytest.mark.parametrize(
        "owner,stage",
        [("main", "process_config-0000"), ("other-policy-actor", "checkpoint_load-0000"), ("main", "old-stage-0000")],
    )
    def test_arbitrary_same_generation_value_is_not_an_owner_proof(self, owner: str, stage: str) -> None:
        """A flat, other-policy or earlier-stage match must not certify the actual consumer."""
        case = "runtime_config/case"
        evidence = _ScopeEvidence(
            actual={case: {f"run/{owner}": {"stages": {stage: {"args": {"hidden_size": 512}}}}}},
            fields={"all": frozenset(), "inference": frozenset()},
            backend_fields=frozenset({"hidden_size"}),
        )
        difference = SnapshotDifference(
            path=[case, "run/inference_controller", "stages", "train_first_step-0000", "args", "hidden_size"],
            before=SnapshotValue(present=True, value=512),
            after=SnapshotValue(present=False, value=None),
        )
        result = evidence.explain(difference=difference)
        assert result["verdict"] == "same-generation-value-found"
        assert result["owner_verified"] is False
        assert result["consumer_verified"] is False
        assert result["witness"][1:4] == [f"run/{owner}", "stages", stage]
        assert difference.before.present and not difference.after.present
