"""Identifier sanitization utilities.

Moved verbatim from ``app/naming.py`` (the audit's §10 "model to clone") and
extended with :func:`sanitize_columns`, which reuses :func:`uniquify` so two
originals differing only in stripped characters can never collapse to the same
column name (fixes §3.2: the old ``str.replace("[^A-Za-z0-9_]", "")`` in
``enhance.py`` produced duplicate columns).

Pure: no pandas, no streamlit, no fastapi. Operates on plain strings / sequences.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

_NON_IDENT_RE = re.compile(r"[^A-Za-z0-9_]+")


def sanitize_identifier(value: str, fallback: str = "wordlist") -> str:
    """Reduce an arbitrary string to a safe identifier ``[A-Za-z0-9_]``.

    Non-identifier runs collapse to a single ``_``; leading/trailing underscores
    are stripped; an empty result falls back to ``fallback``.
    """
    cleaned = _NON_IDENT_RE.sub("_", value).strip("_")
    return cleaned or fallback


def uniquify(value: str, existing: set[str]) -> str:
    """Return ``value`` or ``value_2``, ``value_3``, ... until unique w.r.t. ``existing``.

    Does *not* mutate ``existing``; the caller is expected to add the result.
    """
    if value not in existing:
        return value
    index = 2
    while f"{value}_{index}" in existing:
        index += 1
    return f"{value}_{index}"


def sanitize_columns(columns: Iterable[str]) -> list[str]:
    """Sanitize a sequence of column names, guaranteeing uniqueness.

    Replaces the lossy regex-only sanitization that could create duplicate
    column headers (§3.2). Each name is sanitized with :func:`sanitize_identifier`
    and then :func:`uniquify`-ed against the names already produced, so the
    output length always equals the input length and contains no duplicates.
    """
    seen: set[str] = set()
    result: list[str] = []
    for col in columns:
        sanitized = sanitize_identifier(str(col))
        unique = uniquify(sanitized, seen)
        seen.add(unique)
        result.append(unique)
    return result


__all__ = ["sanitize_columns", "sanitize_identifier", "uniquify"]
