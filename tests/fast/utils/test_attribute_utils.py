import pytest

from tests.fast.fixtures.attribute_fixtures import AttributeProbe

from miles.utils.attribute_utils import has_external_attribute, read_external_attribute
from miles.utils.eval_config import pick_from_args


class TestExternalAttributes:
    @pytest.mark.parametrize("name", ["missing", "absent_property"])
    def test_missing_values_raise_or_return_the_exact_default(self, name: str) -> None:
        """Missing attributes preserve required lookup and arbitrary default identity."""
        probe = AttributeProbe()
        default = object()
        with pytest.raises(AttributeError):
            read_external_attribute(probe, name)
        assert read_external_attribute(probe, name, default) is default
        assert read_external_attribute(probe, name, None) is None
        assert not has_external_attribute(probe, name)

    @pytest.mark.parametrize("name,expected", [("value", None), ("false_value", False)])
    def test_present_values_are_not_replaced_by_defaults(self, name: str, expected: object) -> None:
        """Existing None and false values remain present and retain their values."""
        probe = AttributeProbe()
        assert read_external_attribute(probe, name, object()) is expected
        assert has_external_attribute(probe, name)

    def test_non_attribute_errors_propagate_from_properties(self) -> None:
        """Property failures other than AttributeError are never swallowed."""
        probe = AttributeProbe()
        with pytest.raises(RuntimeError, match="property evaluation failed"):
            read_external_attribute(probe, "broken_property", None)
        with pytest.raises(RuntimeError, match="property evaluation failed"):
            has_external_attribute(probe, "broken_property")

    def test_pick_from_args_keeps_its_first_non_none_contract(self) -> None:
        """The existing picker skips missing and None values while retaining False."""
        assert pick_from_args(AttributeProbe(), ("missing", "value", "false_value")) is False
        assert pick_from_args(AttributeProbe(), ("missing", "value")) is None
