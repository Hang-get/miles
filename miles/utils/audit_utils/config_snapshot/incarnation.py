from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotBinding,
    ConfigSnapshotGeneration,
    ConfigSnapshotInitialization,
    ConfigSnapshotMembership,
    ConfigSnapshotRecord,
)
from miles.utils.audit_utils.config_snapshot.normalizer import normalized_source_name
from miles.utils.audit_utils.process_identity import TrainProcessIdentity
from miles.utils.workers.naming import parse_cell_id


@dataclass(frozen=True)
class SnapshotCaptureGroup:
    name: str
    captures: list[list[ConfigSnapshotRecord]]
    training: bool


def group_snapshot_captures(
    records: list[ConfigSnapshotRecord],
    *,
    bindings: Sequence[ConfigSnapshotBinding],
    initializations: Sequence[ConfigSnapshotInitialization],
) -> list[SnapshotCaptureGroup]:
    by_capture: dict[str, list[ConfigSnapshotRecord]] = defaultdict(list)
    for record in records:
        by_capture[record.context.capture_id].append(record)
    for capture_id, capture in by_capture.items():
        if any(record.context != capture[0].context for record in capture):
            raise ValueError(f"Mixed snapshot contexts for capture {capture_id}")

    membership_by_capture: dict[str, ConfigSnapshotMembership] = {}
    for binding in bindings:
        if binding.capture_id not in by_capture:
            raise ValueError(f"Snapshot binding has no capture: {binding.capture_id}")
        if membership_by_capture.setdefault(binding.capture_id, binding.membership) != binding.membership:
            raise ValueError(f"Conflicting snapshot binding: {binding.capture_id}")
    completed = {frozenset(item.capture_ids) for item in initializations}
    if any(not capture_ids.issubset(by_capture) for capture_ids in completed):
        raise ValueError("Snapshot initialization references missing captures")

    groups: dict[tuple[ConfigSnapshotGeneration, str, str | None], SnapshotCaptureGroup] = {}
    for capture_id, capture in by_capture.items():
        context = capture[0].context
        name = f"{context.name}/{context.deploy_component}/{context.deploy_instance_id}/{normalized_source_name(context.source)}"
        training = isinstance(context.source, TrainProcessIdentity)
        if training:
            if (membership := membership_by_capture.get(capture_id)) is None:
                raise ValueError(f"Unbound training snapshot capture: {capture_id}")
            if context.source.cell_index != parse_cell_id(membership.cell_id).cell_index:
                raise ValueError(f"Snapshot binding cell mismatch: {capture_id}")
            name = f"{name}/{membership.cell_id}"
            key = (context.generation, name, membership.workers_hash)
        else:
            if capture_id in membership_by_capture:
                raise ValueError(f"Non-training snapshot has cell membership: {capture_id}")
            key = (context.generation, name, None)
        groups.setdefault(key, SnapshotCaptureGroup(name=name, captures=[], training=training)).captures.append(
            capture
        )

    used_completions = set()
    for group in groups.values():
        if not group.training:
            continue
        capture_ids = frozenset(capture[0].context.capture_id for capture in group.captures)
        memberships = [membership_by_capture[capture_id] for capture_id in capture_ids]
        if any(membership != memberships[0] for membership in memberships[1:]):
            raise ValueError(f"Ranks disagree on snapshot membership: {group.name}")
        ranks = [capture[0].context.source.rank_within_cell for capture in group.captures]
        if sorted(ranks) != memberships[0].expected_ranks:
            raise ValueError(f"Incomplete or duplicate snapshot ranks: {group.name}: {ranks}")
        if capture_ids not in completed:
            raise ValueError(f"Snapshot incarnation has no completed initialization: {group.name}")
        used_completions.add(capture_ids)
    if used_completions != completed:
        raise ValueError("Snapshot initialization crosses incarnation boundaries")
    return list(groups.values())
