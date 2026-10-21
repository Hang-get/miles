from collections import defaultdict
from collections.abc import Sequence

from pydantic import JsonValue

from miles.utils.audit_utils.config_snapshot.compact_codec import DeltaCodec
from miles.utils.audit_utils.config_snapshot.incarnation import group_snapshot_captures
from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotBinding,
    ConfigSnapshotCase,
    ConfigSnapshotDelta,
    ConfigSnapshotGeneratedValue,
    ConfigSnapshotInitialization,
    ConfigSnapshotPoint,
    ConfigSnapshotProcess,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.normalizer import collect_generated_values, normalize_record
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.test_utils.snapshot import dump_snapshot

_BASE = ConfigSnapshotPoint(stage="process_config", index=0).to_key()


class ConfigSnapshotConverter:
    @classmethod
    def convert(
        cls,
        records: list[ConfigSnapshotRecord],
        *,
        bindings: Sequence[ConfigSnapshotBinding] = (),
        initializations: Sequence[ConfigSnapshotInitialization] = (),
    ) -> ConfigSnapshotCase:
        groups = group_snapshot_captures(records, bindings=bindings, initializations=initializations)
        incarnations: dict[str, list[ConfigSnapshotProcess]] = defaultdict(list)
        processes: dict[str, ConfigSnapshotProcess] = {}
        for group in groups:
            group_records = [record for capture in group.captures for record in capture]
            generated_values = collect_generated_values(records, referenced_records=group_records)
            converted = [
                _convert_process(capture, generated_values=generated_values[capture[0].context.generation])
                for capture in group.captures
            ]
            process = converted[0]
            for other in converted[1:]:
                process = _merge_ranks(existing=process, process=other, name=group.name)
            if group.training:
                incarnations[group.name].append(process)
            elif (existing := processes.get(group.name)) is not None:
                processes[group.name] = _merge_ranks(existing=existing, process=process, name=group.name)
            else:
                processes[group.name] = process
        for name, groups in incarnations.items():
            for index, process in enumerate(sorted(groups, key=dump_snapshot)):
                processes[f"{name}/incarnation-{index:04d}"] = process
        return ConfigSnapshotCase(processes=dict(sorted(processes.items())))


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
    if dump_snapshot(existing.base) != dump_snapshot(process.base) or dump_snapshot(existing.diffs) != dump_snapshot(
        process.diffs
    ):
        raise ValueError(f"Processes disagree on snapshot contents or stages: {name}")
    return process.model_copy(update={"ranks": sorted(set(existing.ranks + process.ranks))})


def _diff_from_base(*, base: JsonValue, samples: dict[str, JsonValue]) -> dict[str, ConfigSnapshotDelta]:
    return {name: DeltaCodec.encode(base=base, actual=sample) for name, sample in samples.items()}
