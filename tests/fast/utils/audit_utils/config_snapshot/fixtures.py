from collections import defaultdict
from dataclasses import dataclass

from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotBinding,
    ConfigSnapshotCase,
    ConfigSnapshotInitialization,
    ConfigSnapshotMembership,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import TrainProcessIdentity


@dataclass
class CaptureMetadata:
    bindings: list[ConfigSnapshotBinding]
    initializations: list[ConfigSnapshotInitialization]


def metadata_for_records(records: list[ConfigSnapshotRecord]) -> CaptureMetadata:
    groups = defaultdict(dict)
    for record in records:
        context = record.context
        if isinstance(context.source, TrainProcessIdentity):
            groups[(context.generation, context.source.to_cell_name())][context.capture_id] = context
    bindings, initializations = [], []
    for captures in groups.values():
        contexts = list(captures.values())
        source = contexts[0].source
        membership = ConfigSnapshotMembership(
            cell_id=f"{source.component}-{source.cell_index:05d}",
            workers_hash="test-workers",
            expected_ranks=sorted({context.source.rank_within_cell for context in contexts}),
        )
        bindings.extend(ConfigSnapshotBinding(capture_id=capture_id, membership=membership) for capture_id in captures)
        initializations.append(ConfigSnapshotInitialization(capture_ids=sorted(captures)))
    return CaptureMetadata(bindings=bindings, initializations=initializations)


def convert_records(records: list[ConfigSnapshotRecord]) -> ConfigSnapshotCase:
    metadata = metadata_for_records(records)
    return ConfigSnapshotConverter.convert(
        records, bindings=metadata.bindings, initializations=metadata.initializations
    )


def write_record(storage: ConfigSnapshotStorage, record: ConfigSnapshotRecord) -> None:
    storage.write(record)
    metadata = metadata_for_records([record])
    for binding in metadata.bindings:
        storage.write_binding(binding)
    for initialization in metadata.initializations:
        storage.write_initialization(initialization)
