import asyncio
import signal
from pathlib import Path

import pytest
from tests.fast.utils.soak.core.archive_fakes import _GatedCollector
from tests.fast.utils.soak.soak_fakes import _at, _observation, _run_context, _wait_until
from tests.utils.soak.core import archive as archive_module
from tests.utils.soak.core.archive import archive_evidence
from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakCollectionClosedEvent,
    SoakEvidenceArchivedEvent,
    SoakObservationEvent,
    file_sha256,
    read_events,
)


def _make_source(root: Path, name: str, files: dict[str, str]) -> Path:
    source = root / name
    source.mkdir(parents=True)
    for relative, content in files.items():
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        (source / relative).write_text(content)
    return source


class TestArchiveEvidence:
    async def test_sources_and_their_discarded_siblings_are_copied_hashed_and_closed(self, tmp_path: Path) -> None:
        """Live and .trash_* event directories are archived with hashes of the archived copies."""
        dumps = tmp_path / "dumps"
        source = _make_source(dumps, "events", {"a.jsonl": "a\n", "nested/b.jsonl": "b\n"})
        _make_source(dumps, ".trash_1", {"c.jsonl": "c\n"})
        (dumps / ".trash_file").write_text("not a directory\n")
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({"training_events": source}, at=_at(0)))

        await archive_evidence(log)

        archived, closed = log.events[-2:]
        assert isinstance(archived, SoakEvidenceArchivedEvent)
        assert isinstance(closed, SoakCollectionClosedEvent)
        root = log.path.parent / "sources" / "training_events"
        assert archived.sources == {"training_events": root / "events"}
        assert archived.missing_sources == []
        assert archived.sha256_of_file == {
            "sources/training_events/.trash_1/c.jsonl": file_sha256(root / ".trash_1" / "c.jsonl"),
            "sources/training_events/events/a.jsonl": file_sha256(root / "events" / "a.jsonl"),
            "sources/training_events/events/nested/b.jsonl": file_sha256(root / "events" / "nested" / "b.jsonl"),
        }
        assert (root / "events" / "nested" / "b.jsonl").read_text() == "b\n"
        assert not (root / ".trash_file").exists()
        assert read_events(log.path) == log.events

    async def test_a_missing_source_is_recorded_as_missing_not_archived(self, tmp_path: Path) -> None:
        """A source that never appeared is listed so checkers refuse to read it."""
        present = _make_source(tmp_path, "present", {"a.jsonl": "a\n"})
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({"training_events": tmp_path / "absent", "other": present}, at=_at(0)))

        await archive_evidence(log)

        archived = log.events[-2]
        assert isinstance(archived, SoakEvidenceArchivedEvent)
        assert archived.missing_sources == ["training_events"]
        assert set(archived.sources) == {"other"}

    async def test_the_latest_run_context_selects_the_archived_sources(self, tmp_path: Path) -> None:
        """Only the sources of the last recorded run context are archived."""
        old = _make_source(tmp_path, "old", {"a.jsonl": "a\n"})
        new = _make_source(tmp_path, "new", {"b.jsonl": "b\n"})
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({"training_events": old}, at=_at(0)))
        log.append(_run_context({"training_events": new}, at=_at(1)))

        await archive_evidence(log)

        archived = log.events[-2]
        assert isinstance(archived, SoakEvidenceArchivedEvent)
        assert archived.sources["training_events"].name == "new"

    async def test_a_log_without_run_context_is_only_closed(self, tmp_path: Path) -> None:
        """Without declared sources collection closes without an archive event."""
        log = EventLog(tmp_path / "events.jsonl")
        log.append(_observation(None, at=_at(0)))

        await archive_evidence(log)

        assert [type(event) for event in log.events] == [SoakObservationEvent, SoakCollectionClosedEvent]

    async def test_changing_the_live_source_after_archiving_keeps_the_evidence_valid(self, tmp_path: Path) -> None:
        """Hashes pin the archived copies, so later writes to the live directory do not matter."""
        source = _make_source(tmp_path, "events", {"a.jsonl": "a\n"})
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({"training_events": source}, at=_at(0)))

        await archive_evidence(log)
        (source / "a.jsonl").write_text("later\n")

        assert read_events(log.path) == log.events

    @pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b"])
    async def test_a_source_name_that_is_not_a_plain_directory_name_is_refused(
        self, tmp_path: Path, name: str, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """A source name cannot place archived files outside its own archive directory."""
        source = _make_source(tmp_path, "events", {"a.jsonl": "a\n"})
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({name: source}, at=_at(0)))

        with pytest.raises(RuntimeError, match="collection failed with exit code 1"):
            await archive_evidence(log)
        assert "Invalid evidence source name" in capfd.readouterr().err
        assert len(log.events) == 1
        assert not (log.path.parent / "sources").exists()

    async def test_a_disk_log_that_differs_from_memory_is_not_closed(self, tmp_path: Path) -> None:
        """Collection refuses to close when the persisted log no longer matches what was recorded."""
        log = EventLog(tmp_path / "events.jsonl")
        log.append(_observation(None, at=_at(0)))
        log.path.write_text("")

        with pytest.raises(AssertionError, match="differs from memory"):
            await archive_evidence(log)
        assert not any(isinstance(event, SoakCollectionClosedEvent) for event in log.events)

    async def test_a_copy_failure_propagates_and_leaves_the_log_open(self, tmp_path: Path) -> None:
        """A failed child copy propagates its exit status without publishing archived or closed evidence."""
        source = _make_source(tmp_path, "events", {"a.jsonl": "a\n"})
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({"training_events": source}, at=_at(0)))
        target = log.path.parent / "sources" / "training_events" / source.name
        target.parent.mkdir(parents=True)
        target.write_text("existing destination")

        with pytest.raises(RuntimeError, match="collection failed with exit code 1"):
            await archive_evidence(log)
        assert target.read_text() == "existing destination"
        assert not (log.path.parent / "archive_manifest.json").exists()
        assert len(log.events) == 1

    async def test_a_collection_exceeding_its_bound_stops_the_child_without_closing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, blocked_collector: _GatedCollector
    ) -> None:
        """A timeout reaps the blocked collector before returning, so it cannot write after collection ends."""
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({"training_events": tmp_path / "events"}, at=_at(0)))
        monkeypatch.setattr(archive_module, "_COLLECTION_TIMEOUT_SECONDS", 0.05)

        with pytest.raises(TimeoutError, match="collection exceeded"):
            await archive_evidence(log)

        assert blocked_collector.started.is_set()
        assert blocked_collector.exitcode_on_close in (-signal.SIGTERM, -signal.SIGKILL)
        assert not (tmp_path / "late-write").exists()
        assert not (log.path.parent / "archive_manifest.json").exists()
        assert len(log.events) == 1

    async def test_a_cancelled_collection_stops_the_child_without_closing(
        self, tmp_path: Path, blocked_collector: _GatedCollector
    ) -> None:
        """Cancellation reaps the running collector before propagating and never closes partial evidence."""
        log = EventLog(tmp_path / "evidence" / "events.jsonl")
        log.append(_run_context({"training_events": tmp_path / "events"}, at=_at(0)))
        task = asyncio.create_task(archive_evidence(log))
        await _wait_until(blocked_collector.started.is_set)

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert blocked_collector.exitcode_on_close in (-signal.SIGTERM, -signal.SIGKILL)
        assert not (tmp_path / "late-write").exists()
        assert not (log.path.parent / "archive_manifest.json").exists()
        assert len(log.events) == 1
