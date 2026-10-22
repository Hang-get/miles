from pathlib import Path

import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_expansion import expand_snapshot_directory
from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_input import _load_json, _load_yaml


class TestUnambiguousInputs:
    @pytest.mark.parametrize(
        "text", ["a: 1\na: 2\n", "config: {lr: 0.1, lr: 0.2}", "base: &b {a: 1, a: 2}\nconfig: {<<: *b}"]
    )
    def test_duplicate_explicit_yaml_keys_are_rejected(self, tmp_path: Path, text: str) -> None:
        """Repeated definitions cannot disappear before strict comparison."""
        (tmp_path / "case.yaml").write_text(text)
        with pytest.raises(ValueError, match="Duplicate explicit YAML key"):
            expand_snapshot_directory(directory=tmp_path)

    def test_explicit_merge_override_and_alias_reuse_remain_legal(self) -> None:
        """Compression merge precedence is distinct from duplicate explicit definitions."""
        value = _load_yaml("base: &base {a: 1}\nconfig: {<<: *base, a: 2}\nother: {<<: *base}\n")
        assert value == {"base": {"a": 1}, "config": {"a": 2}, "other": {"a": 1}}

    def test_duplicate_json_proof_key_is_rejected(self) -> None:
        """Proof and rule documents cannot overwrite a contradictory object member."""
        with pytest.raises(ValueError, match="Duplicate explicit key"):
            _load_json('{"proof": {"match": false, "match": true}}')

    def test_duplicate_text_delta_metadata_is_rejected_before_expansion(self, tmp_path: Path) -> None:
        """Text delta parsing must not discard a conflicting base declaration."""
        (tmp_path / "case.txt").write_text("# miles-snapshot-delta-v1\nbase: one\nbase: two\n")
        with pytest.raises(ValueError, match="Duplicate explicit YAML key"):
            expand_snapshot_directory(directory=tmp_path)

    def test_unused_shared_override_template_is_rejected(self, tmp_path: Path) -> None:
        """A misspelled compact override cannot be silently discarded by the decoder."""
        directory = tmp_path / "runtime_config"
        directory.mkdir()
        (directory / "base.yaml").write_text("templates: {trainer: {args: {}}}")
        (directory / "case.yaml").write_text(
            'shared_overrides: {typo: {set: {"/x": 1}}}\nprocesses: {actor: {template: trainer}}'
        )
        with pytest.raises(AssertionError, match="Unknown shared override template"):
            expand_snapshot_directory(directory=tmp_path)
