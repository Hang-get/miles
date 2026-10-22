from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_sglang_evidence import _split_models


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
