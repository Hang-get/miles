import asyncio
import multiprocessing
import shutil
from multiprocessing.process import BaseProcess
from pathlib import Path

from tests.utils.soak.core.event_log import EventLog
from tests.utils.soak.core.events import (
    SoakCollectionClosedEvent,
    SoakEvidenceArchivedEvent,
    SoakRunContextEvent,
    file_sha256,
    read_events,
)

_COLLECTION_TIMEOUT_SECONDS = 180.0


async def archive_evidence(event_log: EventLog) -> None:
    path = event_log.path
    archived = None
    if contexts := [event for event in event_log.events if isinstance(event, SoakRunContextEvent)]:
        manifest = path.parent / "archive_manifest.json"
        process = multiprocessing.get_context("spawn").Process(
            target=_collect,
            kwargs={"sources": contexts[-1].sources, "destination": path.parent / "sources", "manifest": manifest},
        )
        process.start()
        joining = asyncio.create_task(asyncio.to_thread(process.join, _COLLECTION_TIMEOUT_SECONDS))
        try:
            await asyncio.shield(joining)
            if process.is_alive():
                raise TimeoutError(f"Soak evidence collection exceeded {_COLLECTION_TIMEOUT_SECONDS}s")
            if process.exitcode != 0:
                raise RuntimeError(f"Soak evidence collection failed with exit code {process.exitcode}")
            archived = SoakEvidenceArchivedEvent.model_validate_json(manifest.read_text())
        finally:
            await _close_process(process=process, joining=joining)

    assert (
        read_events(path, require_closed=False) == event_log.events
    ), f"Persisted soak evidence differs from memory: {path}"
    if archived is not None:
        event_log.append(archived)
    event_log.append(SoakCollectionClosedEvent())


async def _close_process(*, process: BaseProcess, joining: asyncio.Task[None]) -> None:
    for stop in (process.terminate, process.kill):
        if process.is_alive():
            stop()
        if joining.done():
            joining = asyncio.create_task(asyncio.to_thread(process.join, 5.0))
        try:
            await asyncio.wait_for(asyncio.shield(joining), timeout=5.0)
        except TimeoutError:
            continue
        if not process.is_alive():
            process.close()
            return
    raise RuntimeError("Soak evidence collector did not stop")


def _collect(*, sources: dict[str, Path], destination: Path, manifest: Path) -> None:
    archived = _archive_sources(sources=sources, destination=destination)
    manifest.write_text(archived.model_dump_json())


def _archive_sources(*, sources: dict[str, Path], destination: Path) -> SoakEvidenceArchivedEvent:
    archived: dict[str, Path] = {}
    missing: list[str] = []
    hashes: dict[str, str] = {}
    for name, source in sources.items():
        assert name and Path(name).name == name and name not in (".", ".."), f"Invalid evidence source name: {name}"
        if not source.is_dir():
            missing.append(name)
            continue
        root = destination / name
        target = root / source.name
        shutil.copytree(source, target)
        for discarded in sorted(source.parent.glob(".trash_*")):
            if discarded.is_dir():
                shutil.copytree(discarded, root / discarded.name)
        archived[name] = target
        for path in sorted(root.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(destination.parent))] = file_sha256(path)
    return SoakEvidenceArchivedEvent(sources=archived, missing_sources=missing, sha256_of_file=hashes)
