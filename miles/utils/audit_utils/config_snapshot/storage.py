import uuid
from dataclasses import dataclass
from pathlib import Path

from miles.utils.audit_utils.config_snapshot.models import (
    ConfigSnapshotBinding,
    ConfigSnapshotInitialization,
    ConfigSnapshotRecord,
)
from miles.utils.file_utils import atomic_write_text


@dataclass(kw_only=True)
class ConfigSnapshotStorage:
    directory: Path

    def write(self, record: ConfigSnapshotRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path=self.directory / f"{uuid.uuid4().hex}.json", text=record.model_dump_json())

    def read(self) -> list[ConfigSnapshotRecord]:
        try:
            paths = sorted(self.directory.iterdir())
        except FileNotFoundError:
            return []

        return [ConfigSnapshotRecord.model_validate_json(path.read_text()) for path in paths if path.suffix == ".json"]

    def write_binding(self, binding: ConfigSnapshotBinding) -> None:
        directory = self.directory / "bindings"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path=directory / f"{uuid.uuid4().hex}.json", text=binding.model_dump_json())

    def read_bindings(self) -> list[ConfigSnapshotBinding]:
        return [
            ConfigSnapshotBinding.model_validate_json(path.read_text())
            for path in sorted((self.directory / "bindings").glob("*.json"))
        ]

    def write_initialization(self, initialization: ConfigSnapshotInitialization) -> None:
        directory = self.directory / "initializations"
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path=directory / f"{uuid.uuid4().hex}.json", text=initialization.model_dump_json())

    def read_initializations(self) -> list[ConfigSnapshotInitialization]:
        return [
            ConfigSnapshotInitialization.model_validate_json(path.read_text())
            for path in sorted((self.directory / "initializations").glob("*.json"))
        ]
