from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.fast.utils.soak.core.archive_fakes import _GatedCollector
from tests.utils.soak.core import archive as archive_module


@pytest.fixture
def blocked_collector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_GatedCollector]:
    collector = _GatedCollector(tmp_path / "late-write")
    monkeypatch.setattr(
        archive_module.multiprocessing, "get_context", lambda _: SimpleNamespace(Process=lambda **kwargs: collector)
    )
    try:
        yield collector
    finally:
        if collector.exitcode_on_close is None:
            if collector.is_alive():
                collector.kill()
                collector.join(timeout=5)
            collector.close()
