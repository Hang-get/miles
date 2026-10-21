import os
import re
from argparse import Namespace
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from miles.utils.audit_utils.config_snapshot.generated_value_registry import generated_value_registry
from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotBinding,
    ConfigSnapshotContext,
    ConfigSnapshotInitialization,
    ConfigSnapshotMembership,
    ConfigSnapshotPoint,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.storage import ConfigSnapshotStorage
from miles.utils.audit_utils.process_identity import ProcessIdentity, TrainProcessIdentity
from miles.utils.env_report.redaction import redact_config_values
from miles.utils.test_utils.snapshot import SNAPSHOT_RECORD_DIR_ENV_VAR, snapshot_values
from miles.utils.workers.naming import parse_cell_id


@dataclass
class _SnapshotState:
    storage: ConfigSnapshotStorage
    context: ConfigSnapshotContext
    membership: ConfigSnapshotMembership | None = None
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))


class ConfigSnapshotDumper:
    _state: ClassVar[_SnapshotState | None] = None

    @classmethod
    def configure(cls, *, args: Namespace, source: ProcessIdentity) -> None:
        cls._state = None
        if args.ci_disable_config_snapshot or not args.ci_test:
            return

        if not (record_directory := os.environ.get(SNAPSHOT_RECORD_DIR_ENV_VAR)):
            return
        if not args.config_snapshot_name:
            raise ValueError("Snapshot name is required")
        cls._state = _SnapshotState(
            storage=ConfigSnapshotStorage(directory=Path(record_directory)),
            context=ConfigSnapshotContext(
                name=args.config_snapshot_name,
                deploy_component=args.deploy_component,
                deploy_instance_id=args.deploy_instance_id or "default",
                source=source,
                run_uuid=args.run_uuid,
            ),
        )

    @classmethod
    def dump(cls, *, stage: str, config: Any) -> None:
        if (state := cls._state) is None:
            return
        if not re.fullmatch(r"[a-z_]+", stage):
            raise ValueError(f"Invalid snapshot stage: {stage!r}")

        index = state.counts[stage]
        state.counts[stage] += 1
        record = ConfigSnapshotRecord(
            context=state.context,
            point=ConfigSnapshotPoint(stage=stage, index=index),
            config=redact_config_values(snapshot_values(config)),
            generated_values=generated_value_registry.read(),
        )
        state.storage.write(record)

    @classmethod
    def enabled(cls) -> bool:
        return cls._state is not None

    @classmethod
    def bind(cls, membership: ConfigSnapshotMembership) -> str | None:
        if (state := cls._state) is None:
            return None
        source = state.context.source
        assert isinstance(source, TrainProcessIdentity), "Only training captures have cell membership"
        assert source.cell_index == parse_cell_id(membership.cell_id).cell_index, "Snapshot cell index mismatch"
        assert source.rank_within_cell in membership.expected_ranks, "Snapshot rank missing from membership"
        if state.membership is not None:
            assert state.membership == membership, "Snapshot capture cannot change incarnation"
            return state.context.capture_id
        state.storage.write_binding(ConfigSnapshotBinding(capture_id=state.context.capture_id, membership=membership))
        state.membership = membership
        return state.context.capture_id

    @classmethod
    def complete_initialization(cls, capture_ids: list[str]) -> None:
        if (state := cls._state) is None:
            return
        state.storage.write_initialization(ConfigSnapshotInitialization(capture_ids=capture_ids))
