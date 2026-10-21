import asyncio
import concurrent.futures
import threading
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
import ray
from tests.fast.utils.fake_ray_ids import fake_ray_node_id

from miles.utils import http_utils
from miles.utils.args.component_rollout import InferenceRuntimeMutState


@pytest.fixture
def isolated_capacity(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(http_utils, "_client_concurrency", 0)
    monkeypatch.setattr(http_utils, "_request_semaphores", {})
    monkeypatch.setattr(http_utils, "_http_clients", {})
    monkeypatch.setattr(http_utils, "_distributed_post_enabled", False)
    monkeypatch.setattr(http_utils, "_post_actors", [])
    monkeypatch.setattr(http_utils, "_post_actor_concurrency", 0)
    monkeypatch.setattr(http_utils, "_post_actor_idx", 0)
    yield


def _args(*, engines: int = 1, distributed: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        inference_runtime_mut_state=InferenceRuntimeMutState(engine_count=engines),
        eval_uses_snapshots=False,
        sglang_server_concurrency=1,
        use_distributed_post=distributed,
        num_gpus_per_node=1,
    )


def test_growth_releases_waiters_before_existing_requests_finish(
    isolated_capacity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Increasing topology capacity admits queued requests without replacing the live client."""

    async def run() -> None:
        args = _args()
        await http_utils.refresh_http_client(args)
        client = http_utils._rollout_client()
        started = [asyncio.Event(), asyncio.Event()]
        finish = asyncio.Event()

        async def send(_client: httpx.AsyncClient, url: str, *_args: Any, **_kwargs: Any) -> str:
            assert _client is client
            started[int(url)].set()
            await finish.wait()
            return url

        monkeypatch.setattr(http_utils, "_post", send)
        first = asyncio.create_task(http_utils.post("0", {}))
        await asyncio.wait_for(started[0].wait(), 1)
        second = asyncio.create_task(http_utils.post("1", {}))
        await asyncio.sleep(0)
        assert not started[1].is_set()
        args.inference_runtime_mut_state.engine_count = 2
        await http_utils.refresh_http_client(args)
        await asyncio.wait_for(started[1].wait(), 1)
        assert not first.done()
        assert http_utils._rollout_client() is client
        assert not client.is_closed
        finish.set()
        assert await asyncio.gather(first, second) == ["0", "1"]
        await client.aclose()

    asyncio.run(run())


def test_shrink_and_zero_topology_preserve_inflight_requests(
    isolated_capacity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shrinking budget waits for active requests and retains one recovery slot at zero engines."""

    async def run() -> None:
        args = _args(engines=2)
        await http_utils.refresh_http_client(args)
        releases = [asyncio.Event() for _ in range(3)]
        starts = [asyncio.Event() for _ in range(3)]

        async def send(_client: Any, url: str, *_args: Any, **_kwargs: Any) -> None:
            index = int(url)
            starts[index].set()
            await releases[index].wait()

        monkeypatch.setattr(http_utils, "_post", send)
        monkeypatch.setattr(http_utils, "_rollout_client", lambda: None)
        tasks = [asyncio.create_task(http_utils.post(str(i), {})) for i in range(2)]
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in starts[:2])), 1)
        args.inference_runtime_mut_state.engine_count = 0
        await http_utils.refresh_http_client(args)
        assert http_utils._client_concurrency == 1
        tasks.append(asyncio.create_task(http_utils.post("2", {})))
        await asyncio.sleep(0)
        assert not starts[2].is_set()
        releases[0].set()
        await tasks[0]
        await asyncio.sleep(0)
        assert not starts[2].is_set()
        assert not tasks[1].done()
        releases[1].set()
        await asyncio.wait_for(starts[2].wait(), 1)
        releases[2].set()
        await asyncio.gather(*tasks)

    asyncio.run(run())


def test_zero_initial_topology_can_recover(isolated_capacity: None) -> None:
    """An initially empty fleet begins admission control when engines appear."""

    async def run() -> None:
        args = _args(engines=0)
        await http_utils.refresh_http_client(args)
        assert http_utils._client_concurrency == 0
        args.inference_runtime_mut_state.engine_count = 3
        await http_utils.refresh_http_client(args)
        assert http_utils._client_concurrency == 3

    asyncio.run(run())


@pytest.mark.parametrize("action", ["post", "get", "delete"])
def test_distributed_failure_reuses_admission_slot_for_local_fallback(
    isolated_capacity: None, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """Distributed fallback and retries stay inside one permit for every supported action."""

    async def run() -> None:
        async def fail(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("synthetic distributed failure")

        actor = SimpleNamespace(do_post=SimpleNamespace(remote=fail))
        monkeypatch.setattr(http_utils, "_post_actors", [actor])
        monkeypatch.setattr(http_utils, "_distributed_post_enabled", True)
        monkeypatch.setattr(http_utils, "_client_concurrency", 1)
        monkeypatch.setattr(http_utils, "_rollout_client", lambda: None)

        async def send(_client: Any, _url: str, _payload: Any, retries: int, **kwargs: Any) -> str:
            assert retries == 3
            assert kwargs["action"] == action
            semaphore = http_utils._request_semaphores[asyncio.get_running_loop()]
            assert semaphore._active == 1
            return "fallback"

        monkeypatch.setattr(http_utils, "_post", send)
        assert await asyncio.wait_for(http_utils.post("url", {}, 3, action=action), 1) == "fallback"
        assert http_utils._request_semaphores[asyncio.get_running_loop()]._active == 0

    asyncio.run(run())


def test_direct_get_shares_budget_and_cancellation_releases_slot(
    isolated_capacity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET waits behind POST and a cancelled local request releases its admission permit."""

    async def run() -> None:
        await http_utils.refresh_http_client(_args())
        started = asyncio.Event()
        got = asyncio.Event()

        async def send(*_args: Any, **_kwargs: Any) -> None:
            started.set()
            await asyncio.Event().wait()

        async def get(_url: str) -> httpx.Response:
            got.set()
            return httpx.Response(200, json={"ok": True}, request=httpx.Request("GET", "http://test"))

        monkeypatch.setattr(http_utils, "_post", send)
        monkeypatch.setattr(http_utils, "_rollout_client", lambda: SimpleNamespace(get=get))
        pending = asyncio.create_task(http_utils.post("url", {}))
        await asyncio.wait_for(started.wait(), 1)
        getter = asyncio.create_task(http_utils.get("url"))
        await asyncio.sleep(0)
        assert not got.is_set()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert await asyncio.wait_for(getter, 1) == {"ok": True}

    asyncio.run(run())


def test_distinct_event_loops_keep_distinct_semaphores(isolated_capacity: None) -> None:
    """Repeated legacy loops never reuse a Condition bound to a different event loop."""
    http_utils.init_http_client(_args())
    seen = []

    async def run() -> None:
        async with http_utils._request_slot():
            seen.append(http_utils._request_semaphores[asyncio.get_running_loop()])

    asyncio.run(run())
    asyncio.run(run())
    assert seen[0] is not seen[1]


def test_distributed_growth_preserves_actors_and_adds_only_missing_capacity(
    isolated_capacity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Poster capacity grows without replacing live actors or recreating them on repeated refreshes."""
    recorded = []

    class Actor:
        def options(self, **options: Any) -> "Actor":
            self.options_value = options
            return self

        def remote(self, *, concurrency: int) -> object:
            assert self.options_value["max_concurrency"] == concurrency
            actor = object()
            recorded.append((actor, concurrency, self.options_value["num_cpus"]))
            return actor

    monkeypatch.setattr(ray, "remote", lambda _cls: Actor())
    monkeypatch.setattr(ray, "nodes", lambda: [{"NodeID": fake_ray_node_id(0), "Alive": True}])
    args = _args(distributed=True)
    http_utils.init_http_client(args)
    original = list(http_utils._post_actors)
    assert len(original) == 1
    original_capacity = recorded[0][1]
    args.inference_runtime_mut_state.engine_count = 7
    http_utils.init_http_client(args)
    grown = list(http_utils._post_actors)
    assert grown[:1] == original
    assert len(grown) * original_capacity >= 7
    assert (len(grown) - 1) * original_capacity < 7
    http_utils.init_http_client(args)
    args.inference_runtime_mut_state.engine_count = 1
    http_utils.init_http_client(args)
    assert http_utils._post_actors == grown
    assert all(cpu == 0.001 for _, _, cpu in recorded)


def test_growth_notifies_waiters_on_another_running_loop(
    isolated_capacity: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A topology refresh wakes a legacy thread's blocked HTTP request on its own event loop."""
    ready: concurrent.futures.Future[None] = concurrent.futures.Future()
    second_started: concurrent.futures.Future[None] = concurrent.futures.Future()
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever)
    finish = asyncio.Event()
    args = _args()
    http_utils.init_http_client(args)

    async def send(_client: Any, url: str, *_args: Any, **_kwargs: Any) -> str:
        if url == "second":
            second_started.set_result(None)
        await finish.wait()
        return url

    async def requests() -> list[str]:
        first = asyncio.create_task(http_utils.post("first", {}))
        await asyncio.sleep(0)
        second = asyncio.create_task(http_utils.post("second", {}))
        await asyncio.sleep(0)
        ready.set_result(None)
        return await asyncio.gather(first, second)

    monkeypatch.setattr(http_utils, "_post", send)
    monkeypatch.setattr(http_utils, "_rollout_client", lambda: None)
    thread.start()
    pending = asyncio.run_coroutine_threadsafe(requests(), loop)

    async def refresh() -> None:
        await asyncio.wait_for(asyncio.wrap_future(ready), 1)
        assert not second_started.done()
        args.inference_runtime_mut_state.engine_count = 2
        await http_utils.refresh_http_client(args)
        await asyncio.wait_for(asyncio.wrap_future(second_started), 1)
        assert not pending.done()
        loop.call_soon_threadsafe(finish.set)
        assert await asyncio.wait_for(asyncio.wrap_future(pending), 1) == ["first", "second"]

    try:
        asyncio.run(refresh())
    finally:
        loop.call_soon_threadsafe(finish.set)
        try:
            pending.result(timeout=2)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2)
            assert not thread.is_alive()
            loop.close()
