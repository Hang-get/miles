"""The rollout executor deletes output store objects per completed rollout, never online."""

import pytest

from miles.ray.rollout import output_store_cleanup
from miles.ray.rollout.output_store_cleanup import OutputStoreCleanup
from miles.utils.output_store.plan import RespRows, Seg, StorePlan, register_dropped_samples, take_dropped_refs
from miles.utils.types import Sample


class _Ref:
    def __init__(self, key: str):
        self.key = key


class _Store:
    def __init__(self, fail: bool = False):
        self.deleted: list[list[str]] = []
        self.fail = fail

    def delete_many(self, refs):
        if self.fail:
            raise RuntimeError("mooncake down")
        self.deleted.append(sorted(ref.key for ref in refs))


def _sample(*keys: str) -> Sample:
    refs = tuple(_Ref(key) for key in keys)
    rows = RespRows((Seg(refs[0], 0, 1),))
    return Sample(rollout_store_plan=StorePlan(fields=(None, None, rows), owned_refs=refs))


@pytest.fixture
def cleanup(monkeypatch):
    store = _Store()
    monkeypatch.setattr(output_store_cleanup, "connect_output_store", lambda args: store)
    take_dropped_refs()
    cleanup = OutputStoreCleanup(args=type("Args", (), {"use_output_store": True})())
    cleanup.connect()
    return cleanup, store


def test_objects_are_deleted_once_their_rollout_is_completed(cleanup):
    cleanup, store = cleanup
    cleanup.record(trainer_model_id=None, rollout_id=3, samples=[[_sample("a", "b")], [_sample("b")]])
    register_dropped_samples([_sample("c")])
    cleanup.record(trainer_model_id=None, rollout_id=4, samples=[_sample("d")])

    cleanup.release_completed(completed_rollout_ids={None: 3})
    assert store.deleted == [["a", "b"]]
    cleanup.release_completed(completed_rollout_ids={None: 3})
    assert store.deleted == [["a", "b"]]
    cleanup.release_completed(completed_rollout_ids={None: 4})
    assert store.deleted == [["a", "b"], ["c", "d"]]


def test_other_models_and_failed_deletes_do_not_stop_training(cleanup):
    cleanup, store = cleanup
    cleanup.record(trainer_model_id="actor", rollout_id=1, samples=[_sample("a")])
    cleanup.release_completed(completed_rollout_ids={"critic": 5})
    assert store.deleted == []
    store.fail = True
    cleanup.release_completed(completed_rollout_ids={"actor": 1})
    assert store.deleted == []


def test_without_the_store_nothing_connects_or_records(monkeypatch):
    def connect(args):
        raise AssertionError("must not connect without --use-output-store")

    monkeypatch.setattr(output_store_cleanup, "connect_output_store", connect)
    take_dropped_refs()
    register_dropped_samples([_sample("a")])
    cleanup = OutputStoreCleanup(args=type("Args", (), {"use_output_store": False})())
    cleanup.connect()
    cleanup.record(trainer_model_id=None, rollout_id=0, samples=[_sample("b")])
    cleanup.release_completed(completed_rollout_ids={None: 0})
    assert [ref.key for ref in take_dropped_refs()] == ["a"]
