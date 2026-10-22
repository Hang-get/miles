import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_comparison_models import SnapshotDifference, SnapshotValue
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_scope_evidence import _ScopeEvidence


class TestSnapshotScopeEvidence:
    @pytest.mark.parametrize(
        "observed,verdict", [(False, "retained-owner-value-verified"), (0, "owner-value-not-captured")]
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
        if verdict == "retained-owner-value-verified":
            assert result["witness"][1] == "run/rollout_executor"
        else:
            assert result["witness"] is None

    def test_controller_backend_removal_requires_a_nested_retained_owner(self) -> None:
        """A dropped controller field has a witness only in an actual backend owner."""
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
        assert result is not None and result["verdict"] == "retained-owner-value-verified"
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
        assert result is not None and result["verdict"] == "owner-value-not-captured"

    @pytest.mark.parametrize(
        "backend,value,verdict",
        [
            ("fsdp", None, "inactive-backend-verified"),
            ("torchtitan", None, "inactive-backend-verified"),
            ("megatron", None, "owner-value-not-captured"),
            ("fsdp", 128, "owner-value-not-captured"),
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
        "debug,verdict", [(True, "disabled-checkpoint-consumer-verified"), (False, "owner-value-not-captured")]
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
