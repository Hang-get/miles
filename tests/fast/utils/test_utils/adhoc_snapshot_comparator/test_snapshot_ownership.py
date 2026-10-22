import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_ownership import _complete_moves, _load_ownership, _Ownership, _scope_moves


class TestSnapshotOwnership:
    @pytest.mark.parametrize("trainer,checkpoint", [(False, False), (True, False), (True, True)])
    def test_declared_ownership_preserves_shared_inputs_and_checkpoint_lifecycle(
        self, trainer: bool, checkpoint: bool
    ) -> None:
        """Only declared fields move, with checkpoint inputs scoped to their load stage."""
        ownership = _Ownership(
            backend_fields={"megatron": frozenset({"lr", "load", "seed"})},
            shared_fields=frozenset({"load", "seed"}),
            trainer_fields=frozenset({"seed"}),
            sglang_fields=frozenset({"sglang_tp_size"}),
            router_fields=frozenset({"router_policy", "worker_urls"}),
            checkpoint_fields=frozenset({"load"}),
        )
        args = {
            "train_backend": "megatron",
            "lr": 1e-6,
            "load": "ckpt",
            "seed": 7,
            "sglang_tp_size": 2,
            "sglang_server_concurrency": 100,
            "router_policy": "cache_aware",
            "worker_urls": ["http://worker"],
            "unknown": False,
        }
        moves = _scope_moves(
            args=args, scope=["case", "config"], ownership=ownership, trainer=trainer, checkpoint=checkpoint
        )
        destinations = {move.source[-1]: move.target[2:] for move in moves}

        target = ["backend"] if trainer else ["raw_megatron", "base_args"]
        assert destinations["lr"] == [*target, "lr"]
        assert destinations["seed"] == [*target, "seed"]
        assert ("load" in destinations) is (not trainer or checkpoint)
        copies = {move.source[-1] for move in moves if move.keep_source}
        assert copies == (set() if trainer else {"load", "seed"})
        assert destinations["sglang_tp_size"] == ["sglang", "base_args", "tp_size"]
        assert destinations["router_policy"] == ["router_args", "policy"]
        assert destinations["worker_urls"] == ["router_args", "worker_urls"]
        assert "unknown" not in destinations
        assert "sglang_server_concurrency" not in destinations
        assert args["unknown"] is False and args["load"] == "ckpt"

    def test_rankless_trainer_controller_uses_backend_ownership(self) -> None:
        """Trainer controllers retain trainer ownership even though they have no distributed ranks."""
        ownership = _Ownership(
            backend_fields={"megatron": frozenset({"lr"})},
            shared_fields=frozenset(),
            trainer_fields=frozenset(),
            sglang_fields=frozenset(),
            router_fields=frozenset(),
            checkpoint_fields=frozenset(),
        )
        snapshots = {
            "runtime_config/case.yaml": {
                "run/trainer_controller_actor": {
                    "ranks": [],
                    "stages": {"process_config-0000": {"args": {"train_backend": "megatron", "lr": 1e-6}}},
                },
                "run/inference_controller": {
                    "ranks": [],
                    "stages": {"process_config-0000": {"args": {"train_backend": "megatron", "lr": 1e-6}}},
                },
            }
        }
        moves = _complete_moves(snapshots=snapshots, ownership=ownership)
        assert len(moves) == 1
        assert moves[0].target[-2:] == ["backend", "lr"]
        assert moves[0].target[1] == "run/trainer_controller_actor"

    def test_shared_sglang_input_is_retained_as_well_as_copied(self) -> None:
        """A typed Miles field also consumed by SGLang must retain both captured owners."""
        ownership = _Ownership(
            backend_fields={"megatron": frozenset()},
            shared_fields=frozenset({"sglang_lora_use_virtual_experts"}),
            trainer_fields=frozenset(),
            sglang_fields=frozenset({"sglang_lora_use_virtual_experts"}),
            router_fields=frozenset(),
            checkpoint_fields=frozenset(),
        )
        moves = _scope_moves(
            args={"train_backend": "megatron", "sglang_lora_use_virtual_experts": False},
            scope=["case"],
            ownership=ownership,
        )
        assert len(moves) == 1 and moves[0].keep_source
        assert moves[0].target == ["case", "sglang", "base_args", "lora_use_virtual_experts"]


def test_removed_dependency_option_retains_historical_ownership() -> None:
    """A retired parser declaration still moves its recorded value without a default."""
    ownership = _load_ownership()
    moves = _scope_moves(
        args={"train_backend": "megatron", "sglang_cuda_graph_tc_compiler": "inductor"},
        scope=["case"],
        ownership=ownership,
    )
    assert len(moves) == 1
    assert moves[0].source == ["case", "sglang_cuda_graph_tc_compiler"]
    assert moves[0].target == ["case", "sglang", "base_args", "cuda_graph_tc_compiler"]
