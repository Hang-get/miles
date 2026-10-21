from collections.abc import Callable

import pytest
from tests.fast.utils.audit_utils.config_snapshot.fixtures import metadata_for_records

from miles.utils.audit_utils.config_snapshot.compact_codec import ConfigSnapshotBases
from miles.utils.audit_utils.config_snapshot.converter import ConfigSnapshotConverter
from miles.utils.audit_utils.config_snapshot.models import ConfigSnapshotGeneratedValue, ConfigSnapshotInitialization
from miles.utils.audit_utils.config_snapshot.serialization import dump_config_snapshot
from miles.utils.test_utils.snapshot import dump_snapshot


class TestSnapshotIncarnations:
    def test_full_groups_sort_by_content_and_keep_identical_restarts(self, make_record: Callable) -> None:
        """Restart ordering and random hashes do not affect contents or erase identical incarnations."""
        records = [
            make_record(run=run, rank=rank, config={"value": value})
            for run, value in enumerate([2, 1, 1])
            for rank in range(2)
        ]
        metadata = metadata_for_records(records)
        records = [
            record.model_copy(
                update={"context": record.context.model_copy(update={"name": "same", "run_uuid": "same-run"})}
            )
            for record in records
        ]
        bindings = [
            binding.model_copy(
                update={
                    "membership": binding.membership.model_copy(
                        update={"workers_hash": binding.capture_id.split("-")[1]}
                    )
                }
            )
            for binding in metadata.bindings
        ]
        result = ConfigSnapshotConverter.convert(records, bindings=bindings, initializations=metadata.initializations)
        reversed_result = ConfigSnapshotConverter.convert(
            list(reversed(records)),
            bindings=list(reversed(bindings)),
            initializations=list(reversed(metadata.initializations)),
        )
        assert dump_snapshot(result) == dump_snapshot(reversed_result)
        assert [process.base["args"]["value"] for process in result.processes.values()] == [1, 1, 2]
        assert [process.ranks for process in result.processes.values()] == [[0, 1]] * 3
        bases = ConfigSnapshotBases(templates={"default": {"args": {"value": 0}}})
        assert dump_config_snapshot(result, bases=bases) == dump_config_snapshot(reversed_result, bases=bases)

    def test_crossed_rank_histories_cannot_pass_as_equal_multisets(self, make_record: Callable) -> None:
        """Matching rank multisets cannot hide disagreement inside each true incarnation."""
        records = [
            make_record(run=run, rank=rank, config={"value": (run + rank) % 2})
            for run in range(2)
            for rank in range(2)
        ]
        metadata = metadata_for_records(records)
        records = [
            record.model_copy(
                update={"context": record.context.model_copy(update={"name": "same", "run_uuid": "same-run"})}
            )
            for record in records
        ]
        bindings = [
            binding.model_copy(
                update={
                    "membership": binding.membership.model_copy(
                        update={"workers_hash": binding.capture_id.split("-")[1]}
                    )
                }
            )
            for binding in metadata.bindings
        ]
        with pytest.raises(ValueError, match="Processes disagree"):
            ConfigSnapshotConverter.convert(records, bindings=bindings, initializations=metadata.initializations)

    @pytest.mark.parametrize(
        "defect",
        ["unbound", "missing_rank", "duplicate_rank", "incomplete_init", "conflicting_binding", "crossed_completion"],
    )
    def test_partial_or_ambiguous_captures_are_rejected(self, make_record: Callable, defect: str) -> None:
        """Only uniquely associated complete rank groups can enter golden comparisons."""
        records = [make_record(rank=rank) for rank in range(2)]
        metadata = metadata_for_records(records)
        if defect == "unbound":
            metadata.bindings.clear()
        elif defect == "missing_rank":
            records.pop()
            metadata.bindings.pop()
            metadata.initializations = [ConfigSnapshotInitialization(capture_ids=[records[0].context.capture_id])]
        elif defect == "duplicate_rank":
            duplicate = records[0].model_copy(
                update={"context": records[0].context.model_copy(update={"capture_id": "duplicate"})}
            )
            records.append(duplicate)
            metadata.bindings.append(metadata.bindings[0].model_copy(update={"capture_id": "duplicate"}))
            metadata.initializations = [
                ConfigSnapshotInitialization(capture_ids=[record.context.capture_id for record in records])
            ]
        elif defect == "incomplete_init":
            metadata.initializations.clear()
        elif defect == "conflicting_binding":
            metadata.bindings.append(
                metadata.bindings[0].model_copy(
                    update={"membership": metadata.bindings[0].membership.model_copy(update={"workers_hash": "other"})}
                )
            )
        else:
            metadata.initializations = [
                ConfigSnapshotInitialization(capture_ids=[record.context.capture_id]) for record in records
            ]
        with pytest.raises(ValueError):
            ConfigSnapshotConverter.convert(
                records, bindings=metadata.bindings, initializations=metadata.initializations
            )

    def test_replayed_metadata_is_idempotent(self, make_record: Callable) -> None:
        """A controller retry may repeat identical associations without inventing another incarnation."""
        records = [make_record(rank=rank) for rank in range(2)]
        metadata = metadata_for_records(records)
        result = ConfigSnapshotConverter.convert(
            records, bindings=metadata.bindings * 2, initializations=metadata.initializations * 2
        )
        assert len(result.processes) == 1

    def test_generated_values_are_checked_within_the_referencing_incarnation(self, make_record: Callable) -> None:
        """Different restart values may reuse a name without creating cross-incarnation conflicts."""
        records = [
            make_record(run=run, config={"save": f"/data/{value}/checkpoint"}).model_copy(
                update={"generated_values": [ConfigSnapshotGeneratedValue(kind="run_id", name="launch", value=value)]}
            )
            for run, value in enumerate(["first", "second"])
        ]
        metadata = metadata_for_records(records)
        records = [
            record.model_copy(
                update={"context": record.context.model_copy(update={"name": "same", "run_uuid": "same-run"})}
            )
            for record in records
        ]
        bindings = [
            binding.model_copy(
                update={
                    "membership": binding.membership.model_copy(
                        update={"workers_hash": binding.capture_id.split("-")[1]}
                    )
                }
            )
            for binding in metadata.bindings
        ]
        result = ConfigSnapshotConverter.convert(records, bindings=bindings, initializations=metadata.initializations)
        assert len(result.processes) == 2
        assert all(
            process.base["args"]["save"] == "/data/$RUN_ID_launch/checkpoint" for process in result.processes.values()
        )
