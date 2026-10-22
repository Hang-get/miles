from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_sglang_evidence import (
    _split_models,
    _verification_status,
)


class TestSglangEvidence:
    def test_owner_split_preserves_all_group_values_and_order(self) -> None:
        """Every original group key survives without synthesizing missing scaling inputs."""
        groups = [
            {"num_gpus": 8, "gpu_offset": 0, "engine_offset": 0, "num_gpus_per_engine": 4, "needs_offload": True},
            {"num_gpus": 4, "gpu_offset": 8, "engine_offset": 2, "num_gpus_per_engine": 2, "needs_offload": False},
        ]
        models, scaling = _split_models(resolved={"models": [{"name": "default", "server_groups": groups}]})
        for index, group in enumerate(groups):
            assert models[0]["server_groups"][index] | scaling["groups"]["default"][index] == group
            assert models[0]["server_groups"][index].keys().isdisjoint(scaling["groups"]["default"][index])


class TestSglangAuditGate:
    def test_scaling_mismatch_is_not_hidden_by_matching_models(self) -> None:
        """Matching model lists cannot certify an incorrect captured GPU layout."""
        status = _verification_status(
            records=[
                {
                    "path": ["case"],
                    "models_match": True,
                    "scaling_observed": True,
                    "scaling_match": False,
                    "counts_status": "verified",
                }
            ]
        )
        assert status["models"] == "verified"
        assert status["scaling"] == "mismatch"

    def test_external_unresolved_counts_remain_unverified(self) -> None:
        """External provider counts are not declared equal to a local GPU-derived count."""
        status = _verification_status(
            records=[
                {
                    "path": ["case"],
                    "models_match": True,
                    "scaling_observed": True,
                    "scaling_match": True,
                    "counts_status": "external-provider-unresolved",
                }
            ]
        )
        assert status["counts"] == "unresolved"

    def test_count_mismatch_fails_even_when_scaling_matches(self) -> None:
        """An incorrect initial cell count cannot inherit the model-list verdict."""
        status = _verification_status(
            records=[
                {
                    "path": ["case"],
                    "models_match": True,
                    "scaling_observed": True,
                    "scaling_match": True,
                    "counts_status": "mismatch-or-missing",
                }
            ]
        )
        assert status["counts"] == "mismatch-or-missing"

    def test_empty_inputs_and_missing_case_owner_do_not_pass(self) -> None:
        """A different case's scaling capture cannot cover an unobserved case."""
        assert _verification_status(records=[])["coverage"] == "no-inputs"
        records = [
            {
                "path": ["captured"],
                "models_match": True,
                "scaling_observed": True,
                "scaling_match": True,
                "counts_status": "verified",
            },
            {
                "path": ["missing"],
                "models_match": True,
                "scaling_observed": False,
                "scaling_match": False,
                "counts_status": "verified",
            },
        ]
        assert _verification_status(records=records)["coverage"] == "scaling-owner-not-captured"

    def test_one_generation_does_not_cover_another_generations_missing_scaling(self) -> None:
        """Each runtime generation needs an observed owner even within the same case."""
        row = {"models_match": True, "scaling_match": True, "counts_status": "verified"}
        records = [
            row
            | {"path": ["runtime_config/case.yaml", "run-0000/all/default/worker_manager"], "scaling_observed": True},
            row
            | {
                "path": ["runtime_config/case.yaml", "run-0000/all/default/inference_controller"],
                "scaling_observed": False,
            },
            row
            | {"path": ["runtime_config/case.yaml", "run-0001/all/default/worker_manager"], "scaling_observed": False},
        ]
        assert _verification_status(records=records[:2])["coverage"] == "verified"
        assert _verification_status(records=records)["coverage"] == "scaling-owner-not-captured"
