from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_coverage_evidence import _unpaired_cases


class TestUnpairedCases:
    def test_records_exact_missing_generation_ranks_and_stages(self) -> None:
        """A differently named run cannot replace the missing counterpart."""
        case = "runtime_config/tests/e2e/example.yaml"
        process = "tests/e2e/example/run-0001/all/default/actor_cell00000"
        observed = {
            process: {
                "ranks": [3, 1],
                "stages": {
                    "process_config-0000": {
                        "args": {"num_rollout": 2, "scaling": {"actor_num_nodes": 2}, "custom_flag": False}
                    },
                    "train_first_step-0000": {"args": {}},
                },
            }
        }
        records = _unpaired_cases(before={"runtime_config/other.yaml": observed}, after={case: observed})
        row = next(item for item in records if item["case"] == case)
        assert row["missing_side"] == "before"
        assert row["required_counterpart"] == [
            {
                "process": process,
                "ranks": [3, 1],
                "stages": ["process_config-0000", "train_first_step-0000"],
                "inputs": {"num_rollout": 2, "scaling": {"actor_num_nodes": 2}, "custom_flag": False},
            }
        ]

    def test_present_null_is_not_a_missing_case(self) -> None:
        """Case presence remains distinct from its value."""
        assert (
            _unpaired_cases(before={"parsed_config/example.yaml": None}, after={"parsed_config/example.yaml": {}})
            == []
        )
