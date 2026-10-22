import pytest

from miles.utils.test_utils.adhoc_snapshot_comparator.snapshot_variant_audit import _diagnostic_views


def test_variant_view_preserves_values_and_does_not_merge_owners() -> None:
    """Diagnostic lookup aliases retain all values and reject owner collisions."""
    old, new = "run/actor", "run/actor/cell/incarnation-0000"
    row = {
        "variant_case": "variant",
        "canonical_case": "canonical",
        "process_identity_correspondences": [{"variant_process": new, "canonical_process": old}],
        "transformed_before_process_correspondences": [{"variant_process": new, "canonical_process": old}],
        "transformed_before_differences": [],
    }
    value = {"ranks": [1, 0], "stages": {"stage": {"flag": False}}}
    views = _diagnostic_views(rows=[row], original={"canonical": {old: value}}, actual={"variant": {new: value}})
    assert views.actual == {"variant": {old: value}}
    assert views.original == {"variant": {old: value}}
    with pytest.raises(AssertionError, match="collision"):
        _diagnostic_views(rows=[row], original={}, actual={"variant": {new: value, old: value}})


def test_before_and_after_aliases_must_identify_the_same_owner() -> None:
    """A before owner cannot silently inherit another actor's evidence."""
    row = {
        "variant_case": "variant",
        "canonical_case": "canonical",
        "process_identity_correspondences": [{"variant_process": "physical", "canonical_process": "actor"}],
        "transformed_before_process_correspondences": [{"variant_process": "physical", "canonical_process": "critic"}],
        "transformed_before_differences": [],
    }
    with pytest.raises(AssertionError, match="owners disagree"):
        _diagnostic_views(rows=[row], original={}, actual={"variant": {}})
