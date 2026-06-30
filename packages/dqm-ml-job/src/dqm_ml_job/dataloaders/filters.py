"""Shared filter utilities for data loaders."""

from collections.abc import Callable
from typing import Any

from dqm_ml_core.utils.matching import has_pattern


def build_filter_condition(
    col: str,
    values: Any,
    wildcard_fn: Callable[[str, list[str]], Any],
    isin_fn: Callable[[str, list[Any]], Any],
    equal_fn: Callable[[str, Any], Any],
) -> Any:
    """Build a filter condition for a column based on the values.

    Args:
        col: Column name to apply the filter on.
        values: Single value or list of values to filter by.
        wildcard_fn: Called with (col, values) when values contain wildcard patterns.
        isin_fn: Called with (col, values) when values is a list without wildcards.
        equal_fn: Called with (col, value) when values is a single value.

    Returns:
        A filter condition suitable for the target backend.
    """
    if isinstance(values, list):
        if any(has_pattern(v) for v in values):
            return wildcard_fn(col, values)
        return isin_fn(col, values)
    return equal_fn(col, values)
