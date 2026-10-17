from pathlib import Path

from tests.e2e.ft.conftest_ft import scenario_trainer_no_failure as scenario
from tests.e2e.ft.conftest_ft.execution import prepare, run_training
from tests.e2e.ft.conftest_ft.modes import resolve_mode
from tests.utils.soak.core.utils import assert_fresh_dump_dir, resolve_dump_dir

from miles.utils.external_utils import command_utils
from miles.utils.workers.types import ClusterBackend


def main() -> None:
    config = command_utils.default_config()
    assert config.cluster_backend is ClusterBackend.RAY
    mode = resolve_mode("kill_train__dp2_cp2__moe_5layer")
    root = Path(resolve_dump_dir("trainer_no_failure_numeric", run_id=config.run_id))
    assert_fresh_dump_dir(root)
    print(f"Diagnostic artifacts: {root}", flush=True)

    for side, builder in (("baseline", scenario._build_baseline_args), ("target", scenario._build_target_args)):
        prepare(mode, config=config)
        dump_dir = root / side
        original = builder(mode, str(dump_dir), True, config)
        diagnostic = original + " --custom-megatron-init-path tests.e2e.ft.conftest_ft.numeric_diagnostic_hook.install "
        (root / f"{side}-original-args.txt").write_text(original)
        (root / f"{side}-diagnostic-args.txt").write_text(diagnostic)
        run_training(
            train_args=diagnostic,
            mode=mode,
            dump_dir=str(dump_dir),
            config=config,
        )

    scenario._compare(str(root), mode)


if __name__ == "__main__":
    main()
