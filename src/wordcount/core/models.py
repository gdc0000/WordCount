"""Interchange dataclasses for the WordCount core.

These are the typed contracts every other core module produces and consumes.
They are the spine of composability: instead of passing around tuples of
dicts (which hid the §2.1 counting bug and forced manual
``normalize_wordlists_for_cache`` tuple-ification), each unit returns a frozen
dataclass with a clear shape.

Hashability (and therefore cacheability) is intentional but **not universal**:

* :class:`CategoryTerms` and :class:`Wordlist` are *hashable* — they are the
  cache key for a run (their content is the user's vocabulary; identical
  vocabulary ⇒ identical analysis). This replaces the legacy
  ``normalize_wordlists_for_cache`` helper.
* :class:`AnalysisConfig`, :class:`DocumentCounts`, and the stats results are
  *frozen but not hashable* — they embed derived structures (a nested-dict
  trie) or a ``DataFrame`` that cannot be hashed. They are rebuilt per cache
  hit / per request and never used as cache keys themselves.

``category_detected`` carries ``{term: occurrences}`` (frequency travels in the
struct), which fixes the §2.2 lossy ``", "``-join round-trip at the source:
plots and the API read frequency directly instead of re-splitting strings.

All classes use ``slots=True`` for a smaller memory footprint (a cheap win for
the memory concern raised in the perf sub-audit, no language change needed).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- #
# Typed errors
#
# Raised by core functions, mapped to HTTP ProblemDetails by api/errors.py.
# Each has a stable ``code`` so the UX layer can branch on it.
# --------------------------------------------------------------------------- #
class WordcountError(Exception):
    """Base class for all core errors. Carries a stable ``code``."""

    code: str = "wordcount_error"

    def __init__(self, message: str = "", *, code: str | None = None) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code


class UnsupportedFormatError(WordcountError):
    """A file extension/mime type the core cannot read."""

    code = "unsupported_format"


class MissingDicTermColumnError(WordcountError):
    """A wordlist file lacks the required ``DicTerm`` column."""

    code = "missing_dicterm_column"


class NoCategoryColumnsError(WordcountError):
    """A wordlist has no category columns (only ``DicTerm``)."""

    code = "no_category_columns"


class NoTextColumnError(WordcountError):
    """A dataset has no column suitable for text analysis."""

    code = "no_text_column"


class AnalysisConfigError(WordcountError):
    """A wordlist/selection cannot yield a valid analysis configuration."""

    code = "analysis_config_error"


# --------------------------------------------------------------------------- #
# Vocabulary (hashable — cache keys)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CategoryTerms:
    """The terms assigned to one category, split by match strategy.

    ``exact_*`` hold whole terms (matched as complete tokens / n-grams);
    ``wildcard_*`` hold prefixes (matched with a trailing-``*`` semantics).
    Single vs. multi distinguishes unigrams from multi-word phrases.
    """

    exact_single: frozenset[str]
    wildcard_single: tuple[str, ...]
    exact_multi: frozenset[str]
    wildcard_multi: tuple[str, ...]

    @classmethod
    def empty(cls) -> CategoryTerms:
        return cls(
            exact_single=frozenset(),
            wildcard_single=(),
            exact_multi=frozenset(),
            wildcard_multi=(),
        )

    @property
    def n_terms(self) -> int:
        """Total distinct terms across all four strategies."""
        return (
            len(self.exact_single)
            + len(self.wildcard_single)
            + len(self.exact_multi)
            + len(self.wildcard_multi)
        )


@dataclass(frozen=True, slots=True)
class Wordlist:
    """One uploaded wordlist, namespaced so its categories stay unique.

    ``categories`` is a sorted tuple of ``(name, CategoryTerms)`` pairs so the
    instance is hashable and its hash is order-independent. Use
    :meth:`from_mapping` for ergonomic construction and
    :meth:`terms_for` / :attr:`category_names` for access.
    """

    namespace: str
    categories: tuple[tuple[str, CategoryTerms], ...]

    @classmethod
    def from_mapping(cls, namespace: str, categories: Mapping[str, CategoryTerms]) -> Wordlist:
        """Build a hashable Wordlist from a plain name->terms mapping."""
        return cls(
            namespace=namespace,
            categories=tuple(sorted(categories.items())),
        )

    @property
    def category_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.categories)

    def terms_for(self, name: str) -> CategoryTerms:
        """Return the terms for a category, raising if absent."""
        for cat_name, terms in self.categories:
            if cat_name == name:
                return terms
        raise KeyError(f"Category {name!r} not in wordlist {self.namespace!r}")

    @property
    def n_categories(self) -> int:
        return len(self.categories)


# --------------------------------------------------------------------------- #
# Analysis configuration (frozen, NOT hashable — derived per run)
# --------------------------------------------------------------------------- #
# The trie is a nested dict built once per run by core.counting.build_analysis_config.
# Its concrete shape is an implementation detail of core/counting.py, hence `Any`
# here; the counting module narrows it locally.
Trie = Any


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Everything :func:`count_document` needs, precomputed once per run.

    Built from a sequence of :class:`Wordlist` objects and a ``max_n`` by
    :func:`core.counting.build_analysis_config`. Frozen for safe sharing across
    worker threads/processes; not hashable because of the trie field. To cache
    analysis results, key on the (hashable) Wordlists + max_n, not on this.
    """

    categories: tuple[str, ...]
    exact_single_lookup: Mapping[str, tuple[str, ...]]  # term -> categories
    exact_multi_lookup: Mapping[str, tuple[str, ...]]  # phrase -> categories
    wildcard_single_trie: Trie
    wildcard_multi_trie: Trie
    required_ngram_lengths: tuple[int, ...]
    max_n: int

    @property
    def has_wildcards(self) -> bool:
        return bool(self.wildcard_single_trie) or bool(self.wildcard_multi_trie)


# --------------------------------------------------------------------------- #
# Per-document result (frozen, NOT hashable — a result, not a key)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class DocumentCounts:
    """The analysis of a single document.

    ``category_counts[cat]`` is the total token occurrences of that category's
    terms in the document (token frequency — the single counting convention
    that fixes §2.1 for exact *and* wildcard, unigrams *and* n-grams).

    ``category_detected[cat]`` is ``{term: occurrences}``: frequency travels in
    the struct so plots/CSV export never need to re-split a joined string
    (fixes §2.2).
    """

    n_tokens: int
    n_types: int
    category_counts: Mapping[str, int]
    category_detected: Mapping[str, Mapping[str, int]]

    @classmethod
    def empty(cls, categories: tuple[str, ...]) -> DocumentCounts:
        return cls(
            n_tokens=0,
            n_types=0,
            category_counts=MappingProxyType({c: 0 for c in categories}),
            category_detected=MappingProxyType({c: MappingProxyType({}) for c in categories}),
        )


# --------------------------------------------------------------------------- #
# Statistics results (frozen, NOT hashable — results)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class PearsonResult:
    """Outcome of ``core.stats.pearson``."""

    col1: str
    col2: str
    coefficient: float
    p_value: float
    n: int


@dataclass(frozen=True, slots=True)
class GroupStats:
    """One group's descriptive stats with a t-based 95% CI (fixes §2.5)."""

    category: str
    n: int
    mean: float
    sem: float
    ci_lower: float
    ci_upper: float


@dataclass(frozen=True, slots=True)
class AnovaResult:
    """Outcome of ``core.stats.anova``.

    ``p_value`` is read by row *name* (``C({cat_var})``), never positionally
    (fixes §2.3). ``significant`` mirrors ``p_value < 0.05``.
    """

    cat_var: str
    num_var: str
    p_value: float
    significant: bool
    table: pd.DataFrame
    group_stats: tuple[GroupStats, ...]
    tukey_rows: tuple[Mapping[str, Any], ...]


__all__ = [
    "AnalysisConfig",
    "AnovaResult",
    "CategoryTerms",
    "DocumentCounts",
    "GroupStats",
    "MissingDicTermColumnError",
    "NoCategoryColumnsError",
    "NoTextColumnError",
    "AnalysisConfigError",
    "PearsonResult",
    "UnsupportedFormatError",
    "WordcountError",
    "Wordlist",
]
