"""Pure dataset & wordlist loaders with typed exceptions.

Replaces ``app/data_io.py`` (audit §4 io-coupling, §5.7, §6.3-at-core). No
``@st.cache_data``, no ``st.error``: known failure modes raise typed
:class:`~wordcount.core.models.WordcountError` subclasses that ``api/errors.py``
will later map to RFC 9457 ProblemDetails. Raw tracebacks are never leaked to
end users by this layer.

``read_dataset`` honors csv/tsv/xls/xlsx and forwards ``dtype``/``usecols`` to
pandas so chunked/typed reads are possible later without changing this unit
(§5.7). ``read_wordlist`` honors the legacy format family
(csv/tsv/txt/dic/dicx/xls/xlsx) and returns a ``dict[str, CategoryTerms]`` —
the ``*`` is stripped here, so ``CategoryTerms.wildcard_*`` hold prefixes
without the star (the trailing-``*`` semantics that ``core/counting.py``
expects). ``merge_wordlists`` returns ``list[Wordlist]`` with ``namespace`` =
sanitized prefix, category names namespaced + uniquified globally (preserves
the audit's §10 praise of the prefix/namespace scheme).

``src`` is flexible: a path (``str``/``Path``) or any file-like with a ``.name``
attribute and a ``.read`` method. The extension is detected from the path
suffix or ``.name``; an explicit ``name`` kwarg overrides detection for
file-likes that carry no name.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import IO, Any

import pandas as pd

from wordcount.core.models import (
    CategoryTerms,
    MissingDicTermColumnError,
    NoCategoryColumnsError,
    UnsupportedFormatError,
    Wordlist,
)
from wordcount.core.naming import sanitize_columns, sanitize_identifier, uniquify

#: A dataset source: a path or a file-like object.
Source = str | PathLike[str] | IO[Any]

#: Cell value (case-insensitive, after strip) marking a term as active in a category.
_ACTIVE_CELL = "X"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _name_of(src: Source, name: str | None) -> str:
    """Resolve a filename for extension detection."""
    if name is not None:
        return name
    if isinstance(src, str | PathLike):
        return str(src)
    return getattr(src, "name", "") or ""


def _suffix(src: Source, name: str | None) -> str:
    return Path(_name_of(src, name)).suffix.lower()


def _stem_of(src: Source, name: str | None) -> str:
    return Path(_name_of(src, name)).stem


def _read_table(src: Source, *, sep: str | None, dtype: Any, usecols: Any) -> pd.DataFrame:
    """Read a delimited table, forwarding dtype/usecols (§5.7)."""
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if usecols is not None:
        kwargs["usecols"] = usecols
    if sep is not None:
        kwargs["sep"] = sep
    return pd.read_csv(src, **kwargs)


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #
def read_dataset(
    src: Source,
    *,
    dtype: Any = None,
    usecols: Any = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Read a dataset (csv/tsv/xls/xlsx) into a :class:`~pandas.DataFrame`.

    Raises :class:`UnsupportedFormatError` for anything else. ``dtype``/``usecols``
    are forwarded to pandas for typed/column-limited reads (§5.7).
    """
    suffix = _suffix(src, name)
    if suffix == ".csv":
        return _read_table(src, sep=None, dtype=dtype, usecols=usecols)
    if suffix == ".tsv":
        return _read_table(src, sep="\t", dtype=dtype, usecols=usecols)
    if suffix in (".xls", ".xlsx"):
        # read_excel does not accept a `sep`; dtype/usecols are supported.
        kwargs: dict[str, Any] = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        if usecols is not None:
            kwargs["usecols"] = usecols
        return pd.read_excel(src, **kwargs)
    raise UnsupportedFormatError(
        f"Unsupported dataset format {suffix!r}: use csv, tsv, xls, or xlsx."
    )


# --------------------------------------------------------------------------- #
# Wordlist loading
# --------------------------------------------------------------------------- #
def _read_wordlist_frame(src: Source, *, name: str | None) -> pd.DataFrame:
    """Read a wordlist frame in any of the legacy formats."""
    suffix = _suffix(src, name)
    if suffix == ".csv":
        return pd.read_csv(src)
    if suffix in (".tsv", ".txt", ".dic"):
        return pd.read_csv(src, sep="\t")
    if suffix == ".dicx":
        return pd.read_csv(src)
    if suffix in (".xls", ".xlsx"):
        return pd.read_excel(src)
    raise UnsupportedFormatError(
        f"Unsupported wordlist format {suffix!r}: use csv, tsv, txt, dic, dicx, xls, or xlsx."
    )


def read_wordlist(src: Source, *, name: str | None = None) -> dict[str, CategoryTerms]:
    """Read one wordlist into ``{category: CategoryTerms}``.

    The frame must have a ``DicTerm`` column (raises
    :class:`MissingDicTermColumnError`); every other column is a category
    (raises :class:`NoCategoryColumnsError` if there are none). Category names
    are sanitized and uniquified within the file via :func:`sanitize_columns`.

    A term is active in a category when that cell is ``X`` (case-insensitive,
    after strip). Terms ending in ``*`` become wildcard prefixes (the ``*`` is
    stripped); multi-word terms go to the ``*_multi`` buckets, single words to
    ``*_single``. Empty terms/prefixes are skipped. Terms are lowercased to
    match :func:`core.tokenize.tokenize`.
    """
    frame = _read_wordlist_frame(src, name=name)

    if "DicTerm" not in frame.columns:
        raise MissingDicTermColumnError("The wordlist must contain a 'DicTerm' column.")

    raw_categories = [c for c in frame.columns if c != "DicTerm"]
    if not raw_categories:
        raise NoCategoryColumnsError("The wordlist has no category columns (only 'DicTerm').")
    categories = sanitize_columns(raw_categories)
    frame.columns = ["DicTerm", *categories]

    buckets: dict[str, dict[str, set[str] | list[str]]] = {
        cat: {
            "exact_single": set(),
            "wildcard_single": [],
            "exact_multi": set(),
            "wildcard_multi": [],
        }
        for cat in categories
    }

    for row in frame.itertuples(index=False):
        raw_term = row.DicTerm
        if pd.isna(raw_term):
            continue
        term = str(raw_term).strip().lower()
        if not term:
            continue
        is_multi = len(term.split()) > 1
        for cat, cell in zip(categories, row[1:], strict=False):
            if str(cell).strip().upper() != _ACTIVE_CELL:
                continue
            bucket = buckets[cat]
            if term.endswith("*"):
                prefix = term[:-1].strip()
                if not prefix:
                    continue
                key = "wildcard_multi" if is_multi else "wildcard_single"
                bucket[key].append(prefix)  # type: ignore[union-attr]
            else:
                key = "exact_multi" if is_multi else "exact_single"
                bucket[key].add(term)  # type: ignore[union-attr]

    return {
        cat: CategoryTerms(
            exact_single=frozenset(b["exact_single"]),
            wildcard_single=tuple(b["wildcard_single"]),
            exact_multi=frozenset(b["exact_multi"]),
            wildcard_multi=tuple(b["wildcard_multi"]),
        )
        for cat, b in buckets.items()
    }


# --------------------------------------------------------------------------- #
# Merge multiple wordlists into namespaced Wordlists
# --------------------------------------------------------------------------- #
def merge_wordlists(
    files: Sequence[Source],
    prefixes: Sequence[str | None] | Mapping[str, str] | None = None,
    *,
    names: Sequence[str | None] | None = None,
) -> list[Wordlist]:
    """Merge multiple wordlists into a list of namespaced :class:`Wordlist`.

    ``prefixes`` is either a parallel sequence (one per file, ``None`` → use the
    file stem) or a mapping keyed by the file's name. ``names`` optionally
    supplies explicit filenames for extension detection / keying when sources
    are file-likes without a ``.name``.

    Each wordlist's ``namespace`` is the sanitized prefix (or stem fallback).
    Category names are namespaced as ``{namespace}_{category}`` and uniquified
    across all wordlists so they never collide globally — the audit's §10
    prefix/namespace scheme, preserved.
    """
    if len(files) == 0:
        return []

    # Normalize prefixes into a parallel list.
    prefix_list: list[str | None]
    if prefixes is None:
        prefix_list = [None] * len(files)
    elif isinstance(prefixes, Mapping):
        prefix_list = [
            prefixes.get(Path(_name_of(f, names[i] if names else None)).name)
            for i, f in enumerate(files)
        ]
    else:
        prefix_list = list(prefixes)
        if len(prefix_list) < len(files):
            prefix_list += [None] * (len(files) - len(prefix_list))

    name_list: list[str | None]
    if names is None:
        name_list = [None] * len(files)
    else:
        name_list = list(names)
        if len(name_list) < len(files):
            name_list += [None] * (len(files) - len(name_list))

    existing: set[str] = set()
    result: list[Wordlist] = []

    for src, prefix, name in zip(files, prefix_list, name_list, strict=False):
        categories_map = read_wordlist(src, name=name)
        stem = _stem_of(src, name) or "wordlist"
        namespace = sanitize_identifier(prefix or stem)

        namespaced: dict[str, CategoryTerms] = {}
        for cat, terms in categories_map.items():
            combined = sanitize_identifier(f"{namespace}_{cat}")
            combined = uniquify(combined, existing)
            existing.add(combined)
            namespaced[combined] = terms

        result.append(Wordlist.from_mapping(namespace, namespaced))

    return result


__all__ = [
    "merge_wordlists",
    "read_dataset",
    "read_wordlist",
]
