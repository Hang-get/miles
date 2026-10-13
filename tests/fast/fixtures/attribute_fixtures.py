class AttributeProbe:
    value = None
    false_value = False

    @property
    def absent_property(self) -> object:
        raise AttributeError("property value is absent")

    @property
    def broken_property(self) -> object:
        raise RuntimeError("property evaluation failed")
