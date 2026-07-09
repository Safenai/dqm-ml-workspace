#!/usr/bin/env python3
"""Generate JSON Schema v2 from the pydantic JobConfig model."""

import json
import re
import sys
from pathlib import Path
from dqm_ml_core.models.config import JobConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE_SRC = REPO_ROOT / "packages" / "dqm-ml-core" / "src"

if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

OUT = REPO_ROOT / "docs" / "schema" / "config.json"

# Ordered list of JSON Schema keywords for consistent display.
_SCHEMA_KEY_ORDER = [
    "title",
    "description",
    "type",
    "const",
    "enum",
    "default",
    "examples",
    "$schema",
    "$id",
    "$ref",
    "anyOf",
    "oneOf",
    "allOf",
    "items",
    "prefixItems",
    "additionalProperties",
    "properties",
    "required",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "$defs",
]

# Desired order of property names at the root "properties" block.
_ROOT_PROPERTIES = [
    "dataloaders",
    "features",
    "metrics",
    "gap",
    "storage",
    "compute",
    "errors",
]


def _clean_description(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned if cleaned else None


def _walk_schema(schema: dict) -> None:
    for key, value in schema.items():
        if key == "description" and isinstance(value, str):
            schema[key] = _clean_description(value)
        elif isinstance(value, dict):
            _walk_schema(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    _walk_schema(item)


def _reorder_root_properties(props: dict[str, object]) -> dict[str, object]:
    """Reorder root-level ``properties`` keys according to _ROOT_PROPERTIES."""
    ordered: dict[str, object] = {}
    for pk in _ROOT_PROPERTIES:
        if pk in props:
            ordered[pk] = props.pop(pk)
    for pk in sorted(props):
        ordered[pk] = props[pk]
    return ordered


def _reorder_dict(obj: dict[str, object], is_root: bool) -> dict[str, object]:
    """Recursively reorder keys of a single dict according to schema conventions."""
    result: dict[str, object] = {}
    for k, v in obj.items():
        if k == "properties" and is_root:
            props = {pk: _reorder(pv) for pk, pv in v.items()}
            result[k] = _reorder_root_properties(props)
        else:
            result[k] = _reorder(v)

    ordered: dict[str, object] = {}
    for k in _SCHEMA_KEY_ORDER:
        if k in result:
            ordered[k] = result[k]
    for k in sorted(result):
        if k not in _SCHEMA_KEY_ORDER:
            ordered[k] = result[k]
    return ordered


def _reorder(obj: object, is_root: bool = False) -> object:
    """Recursively reorder dict keys and root-level ``properties``."""
    if isinstance(obj, dict):
        return _reorder_dict(obj, is_root)
    if isinstance(obj, list):
        return [_reorder(item) for item in obj]
    return obj


def main() -> None:
    schema = JobConfig.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "TODO"
    schema["title"] = "DQM-ML Job Configuration"
    _walk_schema(schema)

    schema = _reorder(schema, is_root=True)  # type: ignore[assignment]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(schema, indent=2) + "\n")

    print(f"Schema generated at {OUT}")


if __name__ == "__main__":
    main()
