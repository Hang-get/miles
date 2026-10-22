import hashlib
from pathlib import Path

import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_capture_recovery import _verify_replayed_records


def test_historical_replay_rejects_changed_or_extra_raw_records(tmp_path: Path) -> None:
    """Historical normalization evidence binds every original record byte and path."""
    path = tmp_path / "record.json"
    path.write_text('{"value": false}')
    expected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()}
    _verify_replayed_records(files=[path], expected=expected)
    path.write_text('{"value": 0}')
    with pytest.raises(AssertionError, match="exact original record inventory"):
        _verify_replayed_records(files=[path], expected=expected)
    with pytest.raises(AssertionError, match="exact original record inventory"):
        _verify_replayed_records(files=[], expected=expected)
