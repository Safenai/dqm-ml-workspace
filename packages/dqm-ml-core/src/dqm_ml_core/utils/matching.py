"""Wildcard pattern matching utilities for config field resolution.

Provides fnmatch-based expansion of ``include`` / ``exclude`` patterns
against a list of available strings (column names, split values, etc.).
"""

import fnmatch


def has_pattern(s: str) -> bool:
    """Return ``True`` if the string contains any fnmatch wildcard character."""
    return any(ch in s for ch in ("*", "?", "["))


def resolve_patterns(patterns: list[str], available: list[str]) -> list[str]:
    """Expand *patterns* against *available* strings using fnmatch.

    Literal (non-wildcard) patterns are emitted as-is.  Wildcard patterns
    are matched against *available* via ``fnmatch.filter``.  Order from
    *patterns* is preserved (first occurrence wins) and duplicates are
    removed.

    Parameters
    ----------
    patterns:
        List of patterns that may contain ``*``, ``?`` or ``[...]``.
    available:
        The set of concrete strings to match against.

    Returns
    -------
        A deduplicated, ordered list of matching concrete strings.
    """
    result: list[str] = []
    for p in patterns:
        if has_pattern(p):
            result.extend(fnmatch.filter(available, p))
        else:
            result.append(p)
    return list(dict.fromkeys(result))


def resolve_include_exclude(
    include: list[str] | None,
    exclude: list[str] | None,
    available: list[str],
) -> list[str]:
    """Resolve (include, exclude) pattern lists against *available* strings.

    1. If *include* is ``None``, every item in *available* is a candidate.
       Otherwise, *include* patterns are expanded via :func:`resolve_patterns`.
    2. Any candidate matching an *exclude* pattern is removed (exact match
       for literal strings, ``fnmatch.filter`` for wildcard patterns).

    Parameters
    ----------
    include:
        Include patterns (``None`` = all).
    exclude:
        Exclude patterns (``None`` = nothing excluded).
    available:
        The set of concrete strings to match against.

    Returns
    -------
        The filtered list (order preserved).
    """
    candidates = resolve_patterns(include, available) if include else list(available)

    if not exclude:
        return candidates

    removed: set[str] = set()
    for p in exclude:
        if has_pattern(p):
            removed.update(fnmatch.filter(candidates, p))
        else:
            removed.add(p)

    return [c for c in candidates if c not in removed]
