from typing import Any


def read_external_attribute(obj: object, attr: str, *default: Any) -> Any:
    return getattr(obj, attr, *default)  # config-access-exempt: attribute selected at runtime from attr


def has_external_attribute(obj: object, attr: str) -> bool:
    try:
        read_external_attribute(obj, attr)
    except AttributeError:
        return False
    return True
