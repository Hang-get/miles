from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_runtime_evidence import _before_trainer_cells


def test_startup_count_uses_critic_gpu_shape_and_all_parallel_dimensions() -> None:
    """Replaying the old formula must not silently use actor resources for a critic."""
    args = {
        "indep_dp": True,
        "critic_num_nodes": 3,
        "critic_num_gpus_per_node": 8,
        "actor_num_nodes": 9,
        "actor_num_gpus_per_node": 8,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 3,
        "context_parallel_size": 2,
    }
    assert _before_trainer_cells(args=args, role="critic") == 2
    assert _before_trainer_cells(args=args, role="actor") == 6
    assert _before_trainer_cells(args={"indep_dp": False}, role="actor") == 1
