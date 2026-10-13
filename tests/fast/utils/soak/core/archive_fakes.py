from multiprocessing import get_context
from multiprocessing.context import SpawnProcess
from multiprocessing.synchronize import Event
from pathlib import Path


class _GatedCollector(SpawnProcess):
    def __init__(self, marker: Path) -> None:
        context = get_context("spawn")
        self.started = context.Event()
        self.release = context.Event()
        self.exitcode_on_close: int | None = None
        super().__init__(
            target=_write_after_release,
            kwargs={"started": self.started, "release": self.release, "marker": marker},
        )

    def start(self) -> None:
        super().start()
        assert self.started.wait(timeout=20), "Collector did not reach the blocked write"

    def close(self) -> None:
        self.exitcode_on_close = self.exitcode
        super().close()


def _write_after_release(*, started: Event, release: Event, marker: Path) -> None:
    started.set()
    assert release.wait(timeout=30), "Collector was not stopped or released"
    marker.write_text("late write")
