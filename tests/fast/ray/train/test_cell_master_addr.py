from pathlib import Path

import pytest
import ray
from tests.fast.ray.train import conftest as train_conftest
from tests.fast.ray.train.conftest import RecordingHealthChecker, get_raw_actor_handles, make_cell, make_indep_dp_info

from miles.utils.workers.worker_handle import WorkerUnreachableError
from miles.utils.workers.worker_spec import HostAndPort

pytestmark = pytest.mark.asyncio


def _calls_of(cell, method: str) -> list:
    return [
        [call for call in ray.get(handle.get_calls.remote()) if call[0] == method]
        for handle in get_raw_actor_handles(cell)
    ]


class TestMasterAddrConfiguration:
    async def test_every_worker_is_told_the_master_address_before_init(self):
        """Workers rendezvous on the address the worker manager allocated for the cell."""
        cell = make_cell(0, actor_count=2)

        await cell.init(ckpt_load_config=None, indep_dp_info=make_indep_dp_info(quorum_id=0), indep_dp_store_addr=None)

        for [call] in _calls_of(cell, "configure_master_addr_and_port"):
            assert call[2] == {"master_addr": "10.0.0.1", "master_port": 20000}

    async def test_the_master_address_is_configured_before_the_process_group_is_built(self):
        """A rank that runs init first would build the process group without the address."""
        cell = make_cell(0, actor_count=1)

        await cell.init(ckpt_load_config=None, indep_dp_info=make_indep_dp_info(quorum_id=0), indep_dp_store_addr=None)

        methods = [call[0] for call in ray.get(get_raw_actor_handles(cell)[0].get_calls.remote())]
        assert methods.index("configure_master_addr_and_port") < methods.index("init")

    async def test_every_worker_rendezvous_on_the_first_workers_unbracketed_endpoint(self):
        """Ranks must share one endpoint, and torch.distributed rejects the URI brackets of an IPv6 host."""
        train_conftest.fake_worker_manager.master_addr_per_worker = [
            HostAndPort(host="[fe80::a]", port=21001),
            HostAndPort(host="[fe80::b]", port=21002),
        ]
        cell = make_cell(0, actor_count=2)

        await cell.init(ckpt_load_config=None, indep_dp_info=make_indep_dp_info(quorum_id=0), indep_dp_store_addr=None)

        for [call] in _calls_of(cell, "configure_master_addr_and_port"):
            assert call[2] == {"master_addr": "fe80::a", "master_port": 21001}


class _FailingConfigureWorkerHandle:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.kill_self_count: int = 0
        self.wait_dead_count: int = 0

    async def configure_master_addr_and_port(self, **_kwargs) -> None:
        self.calls.append("configure_master_addr_and_port")
        raise WorkerUnreachableError("worker died before rendezvous")

    async def init(self, **_kwargs) -> None:
        self.calls.append("init")

    async def kill_self(self) -> None:
        self.kill_self_count += 1

    async def wait_dead(self, *, timeout: float) -> None:
        self.wait_dead_count += 1


class TestMasterAddrConfigurationFailure:
    async def test_a_failed_rendezvous_aborts_before_init_and_tears_the_whole_cell_down(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        """A rank that never learned the address would hang the others in the process-group build."""
        checker = RecordingHealthChecker()
        cell = make_cell(0, actor_count=2, health_checker=checker)
        handles = [_FailingConfigureWorkerHandle(), _FailingConfigureWorkerHandle()]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)

        with pytest.raises(WorkerUnreachableError):
            await cell.init(
                ckpt_load_config=None,
                indep_dp_info=make_indep_dp_info(quorum_id=0),
                indep_dp_store_addr=None,
            )

        assert [handle.calls for handle in handles] == [["configure_master_addr_and_port"]] * 2
        assert [handle.kill_self_count for handle in handles] == [1, 1]
        assert not cell.is_alive
        assert cell.is_errored
        assert checker.start_count == 0


class TestSnapshotInitializationBoundary:
    @pytest.mark.parametrize("enabled", [False, True])
    @pytest.mark.parametrize("fails", [False, True])
    async def test_existing_rendezvous_rpc_carries_membership_and_only_success_records_completion(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, enabled: bool, fails: bool
    ) -> None:
        """Cell initialization binds the observed hash without an extra RPC or false completion."""
        from argparse import Namespace

        from tests.fast.ray.train.conftest import SnapshotWorkerHandle

        from miles.utils.audit_utils.config_snapshot.dumper import ConfigSnapshotDumper
        from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
        from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
        from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR

        monkeypatch.setattr(ConfigSnapshotDumper, "_state", None)
        monkeypatch.setenv(SNAPSHOT_RECORD_DIR_ENV_VAR, str(tmp_path))
        args = Namespace(
            ci_test=True,
            ci_disable_config_snapshot=not enabled,
            config_snapshot_name="cell-init",
            deploy_component="all",
            deploy_instance_id=None,
            run_uuid="test-run",
        )
        ConfigSnapshotDumper.configure(args=args, source=SimpleProcessIdentity(component="main"))
        cell = make_cell(0, actor_count=2)
        handles = [SnapshotWorkerHandle(rank=rank, fail_init=fails) for rank in range(2)]
        monkeypatch.setattr(cell, "_get_worker_handles", lambda: handles)
        if fails:
            with pytest.raises(RuntimeError, match="initialization failed"):
                await cell.init(ckpt_load_config=None, indep_dp_info=make_indep_dp_info(), indep_dp_store_addr=None)
        else:
            await cell.init(ckpt_load_config=None, indep_dp_info=make_indep_dp_info(), indep_dp_store_addr=None)
        assert all(handle.calls == ["configure_master_addr_and_port", "init"] for handle in handles)
        if enabled:
            assert all(handle.membership.workers_hash == cell.workers_hash for handle in handles)
            assert all(handle.membership.expected_ranks == [0, 1] for handle in handles)
        else:
            assert all(handle.membership is None for handle in handles)
            assert list(tmp_path.iterdir()) == []
        completed = ConfigSnapshotStorage(directory=tmp_path).read_initializations()
        assert len(completed) == int(enabled and not fails)
        if completed:
            assert completed[0].capture_ids == ["capture-0", "capture-1"]
