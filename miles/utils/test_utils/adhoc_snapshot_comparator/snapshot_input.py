import json
from typing import Any

import yaml
from yaml.nodes import MappingNode


def _load_json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_unique_pairs)


def _load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_UniqueKeyLoader)


def _load_yaml_all(text: str) -> list[Any]:
    return list(yaml.load_all(text, Loader=_UniqueKeyLoader))


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate explicit key: {key!r}")
        result[key] = value
    return result


class _UniqueKeyLoader(yaml.SafeLoader):
    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._checked_mappings: set[int] = set()

    def flatten_mapping(self, node: MappingNode) -> None:
        if id(node) not in self._checked_mappings:
            self._checked_mappings.add(id(node))
            keys = set()
            for key_node, _ in node.value:
                key = "<<" if key_node.tag == "tag:yaml.org,2002:merge" else self.construct_object(key_node)
                if key in keys:
                    raise ValueError(f"Duplicate explicit YAML key: {key!r}")
                keys.add(key)
        super().flatten_mapping(node)
