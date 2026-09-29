import difflib
from collections import defaultdict

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotCase,
    ConfigSnapshotGeneratedValue,
    ConfigSnapshotPoint,
    ConfigSnapshotProcess,
    ConfigSnapshotProcessDiff,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.normalizer import (
    collect_generated_values,
    normalize_record,
    normalized_source_name,
)
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.test_utils.snapshot import dump_snapshot

_BASE = ConfigSnapshotPoint(stage="process_config", index=0).to_key()


class ConfigSnapshotConverter:
    @classmethod
    def convert(cls, records: list[ConfigSnapshotRecord]) -> ConfigSnapshotCase:
        generated_values = collect_generated_values(records)
        by_capture: dict[str, list[ConfigSnapshotRecord]] = defaultdict(list)
        for record in records:
            by_capture[record.context.capture_id].append(record)

        processes: dict[str, ConfigSnapshotProcess] = {}
        for capture_records in by_capture.values():
            context = capture_records[0].context
            name = (
                f"{context.name}/{context.deploy_component}/{context.deploy_instance_id}/"
                f"{normalized_source_name(context.source)}"
            )
            process = _convert_process(
                capture_records,
                generated_values=generated_values[context.generation],
            )
            if (existing := processes.get(name)) is not None:
                process = _merge_ranks(existing=existing, process=process, name=name)
            processes[name] = process
        return ConfigSnapshotCase(processes=_compress_process_bases(processes))


def _compress_process_bases(
    processes: dict[str, ConfigSnapshotProcess],
) -> dict[str, ConfigSnapshotProcess | ConfigSnapshotProcessDiff]:
    bases: dict[str, tuple[str, ConfigSnapshotProcess]] = {}
    result: dict[str, ConfigSnapshotProcess | ConfigSnapshotProcessDiff] = {}
    for name, process in sorted(processes.items()):
        source = name.rsplit("/", maxsplit=1)[-1]
        if source not in bases:
            bases[source] = (name, process)
            result[name] = process
            continue

        base_name, base = bases[source]
        compressed = ConfigSnapshotProcessDiff(
            ranks=process.ranks,
            base_ref=base_name,
            base_diff="".join(
                difflib.unified_diff(
                    dump_snapshot(base.base).splitlines(keepends=True),
                    dump_snapshot(process.base).splitlines(keepends=True),
                    fromfile=base_name,
                    tofile=name,
                    n=0,
                )
            ),
            diffs=process.diffs,
        )
        result[name] = compressed if len(dump_snapshot(compressed)) < len(dump_snapshot(process)) else process
    return result


def _convert_process(
    records: list[ConfigSnapshotRecord],
    *,
    generated_values: list[ConfigSnapshotGeneratedValue],
) -> ConfigSnapshotProcess:
    context = records[0].context
    samples: dict[str, JsonValue] = {}
    for record in sorted(records, key=lambda record: (record.point.stage, record.point.index)):
        if record.context != context:
            raise ValueError(f"Mixed snapshot contexts for capture {context.capture_id}")
        name = record.point.to_key()
        if name in samples:
            raise ValueError(f"Duplicate snapshot stage for capture {context.capture_id}: {name}")
        samples[name] = normalize_record(record, generated_values=generated_values)
    if _BASE not in samples:
        raise ValueError(f"Missing {_BASE} for capture {context.capture_id}")

    base = samples.pop(_BASE)
    ranks = [context.source.rank_within_cell] if isinstance(context.source, TrainProcessIdentity) else []
    return ConfigSnapshotProcess(ranks=ranks, base=base, diffs=_diff_from_base(base=base, samples=samples))


def _merge_ranks(
    *, existing: ConfigSnapshotProcess, process: ConfigSnapshotProcess, name: str
) -> ConfigSnapshotProcess:
    if dump_snapshot(existing.base) != dump_snapshot(process.base) or existing.diffs != process.diffs:
        raise ValueError(f"Processes disagree on snapshot contents or stages: {name}")
    return process.model_copy(update={"ranks": sorted(set(existing.ranks + process.ranks))})


def _diff_from_base(*, base: JsonValue, samples: dict[str, JsonValue]) -> dict[str, str]:
    base_lines = dump_snapshot(base).splitlines(keepends=True)
    return {
        name: "".join(
            difflib.unified_diff(
                base_lines, dump_snapshot(sample).splitlines(keepends=True), fromfile=_BASE, tofile=name, n=0
            )
        )
        for name, sample in samples.items()
    }
