"""Document counting — the hot path, with the §2.1 fix.

This is the core of the analysis: turn a document + an :class:`AnalysisConfig`
into a :class:`DocumentCounts`, and turn a sequence of documents into the
analysis :class:`~pandas.DataFrame` that :mod:`core.enhance` consumes.

**§2.1 fix — the single counting convention.** Exact *and* wildcard matches,
for unigrams *and* n-grams, all count **token occurrences** (frequencies),
not types. Before, the exact branch iterated unique tokens and did ``+= 1``
(types) while the wildcard branch did ``+= occurrences`` (tokens), so
``"happy happy happy"`` counted 3 for ``happ*`` but 1 for ``happy``. Now both
count 3. ``category_detected[cat][term]`` carries the occurrence count, so
plots/export never re-split a joined string (fixes §2.2 at the source).

The trie (``_build_prefix_trie`` / ``_match_prefix_categories``) and the
reverse-lookup build are ported **unchanged in algorithm** from the ``ace8366``
refactor of ``app/text_analysis.py`` — the audit (§5.2/§5.3, §10) praised them
as the right shape. They now consume :class:`Wordlist` and return
:class:`AnalysisConfig` instead of ad-hoc dicts.

**§5.4 — progress is a callback.** ``analyze_documents(..., progress=cb)``
calls ``cb(completed, total)``; the UI, CLI, and API feed different closures.
The legacy dead ``progress_bar=None`` path is gone by design.

**§5.5 — parallelism is injected.** ``executor=`` accepts any
``concurrent.futures.Executor`` (Thread/Process). ``count_document`` is pure
and module-level so it is picklable; ``AnalysisConfig`` is frozen and safe to
share. (``ThreadPoolExecutor`` shares the config by reference with no pickling;
``ProcessPoolExecutor`` requires the config to be picklable, which holds for its
plain-dict trie and ``MappingProxyType`` lookups on Python 3.13+.)

The dead ``count_words`` from the legacy module is **not** ported.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import pandas as pd

from wordcount.core.models import AnalysisConfig, DocumentCounts, Wordlist
from wordcount.core.tokenize import DEFAULT_MAX_N, MIN_NGRAM_LENGTH, generate_ngrams, tokenize

#: Trie node key marking "the prefixes ending here belong to these categories".
#: Ported verbatim from the ``ace8366`` refactor.
_TRIE_TERMINAL = "_categories_"

Trie = dict[str, Any]


# --------------------------------------------------------------------------- #
# Trie (ported unchanged from app/text_analysis.py)
# --------------------------------------------------------------------------- #
def _build_prefix_trie(prefix_to_categories: Mapping[str, Sequence[str]]) -> Trie:
    """Build a char-wise trie mapping each prefix to the categories it serves.

    Ported unchanged from ``app/text_analysis.py:_build_prefix_trie``.
    """
    trie: Trie = {}
    for prefix, categories in prefix_to_categories.items():
        node: dict[str, Any] = trie
        for char in prefix:
            node = node.setdefault(char, {})
        node.setdefault(_TRIE_TERMINAL, []).extend(categories)
    return trie


def _match_prefix_categories(term: str, trie: Trie) -> list[str]:
    """Return every category whose prefix is a *character* prefix of ``term``.

    Walks the trie along the term's characters, collecting terminal category
    lists at every node. Ported unchanged from
    ``app/text_analysis.py:_match_prefix_categories``.
    """
    if not trie:
        return []
    node: dict[str, Any] = trie
    matched: list[str] = []
    for char in term:
        child = node.get(char)
        if child is None:
            break
        node = child
        matched.extend(node.get(_TRIE_TERMINAL, ()))
    return matched


# --------------------------------------------------------------------------- #
# Config build (cached on hashable Wordlists + max_n — NOT on the config itself)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=128)
def _build_analysis_config_cached(wordlists: tuple[Wordlist, ...], max_n: int) -> AnalysisConfig:
    """Cached core of :func:`build_analysis_config`.

    Keyed on the hashable ``(wordlists, max_n)`` tuple. Equal wordlists produce
    equal hashes ⇒ cache hit ⇒ the *same* frozen :class:`AnalysisConfig` object
    is returned across runs with identical vocabulary.
    """
    category_order: list[str] = []
    exact_single_lookup: dict[str, list[str]] = defaultdict(list)
    exact_multi_lookup: dict[str, list[str]] = defaultdict(list)
    wildcard_single_p2c: dict[str, list[str]] = defaultdict(list)
    wildcard_multi_p2c: dict[str, list[str]] = defaultdict(list)
    required_ngram_lengths: set[int] = set()

    for wl in wordlists:
        for cat_name, terms in wl.categories:
            category_order.append(cat_name)

            for term in terms.exact_single:
                exact_single_lookup[term].append(cat_name)

            for term in terms.exact_multi:
                exact_multi_lookup[term].append(cat_name)
                length = len(term.split())
                if MIN_NGRAM_LENGTH <= length <= max_n:
                    required_ngram_lengths.add(length)

            for prefix in terms.wildcard_single:
                if prefix:
                    wildcard_single_p2c[prefix].append(cat_name)

            for prefix in terms.wildcard_multi:
                if not prefix:
                    continue
                wildcard_multi_p2c[prefix].append(cat_name)
                # A multi-wildcard prefix of L words (L>=1: "new *" strips to
                # the 1-word prefix "new") matches n-grams of orders L..max_n —
                # the prefix is the first L words of the joined phrase. Unigram
                # orders are excluded (unigrams are matched separately).
                length = len(prefix.split())
                if 1 <= length <= max_n:
                    required_ngram_lengths.update(range(max(MIN_NGRAM_LENGTH, length), max_n + 1))

    categories = tuple(dict.fromkeys(category_order))

    return AnalysisConfig(
        categories=categories,
        exact_single_lookup=MappingProxyType(
            {t: tuple(cats) for t, cats in exact_single_lookup.items()}
        ),
        exact_multi_lookup=MappingProxyType(
            {t: tuple(cats) for t, cats in exact_multi_lookup.items()}
        ),
        wildcard_single_trie=_build_prefix_trie(wildcard_single_p2c),
        wildcard_multi_trie=_build_prefix_trie(wildcard_multi_p2c),
        required_ngram_lengths=tuple(sorted(required_ngram_lengths)),
        max_n=max_n,
    )


def build_analysis_config(
    wordlists: Sequence[Wordlist], max_n: int = DEFAULT_MAX_N
) -> AnalysisConfig:
    """Precompute everything :func:`count_document` needs, once per run.

    Accepts any sequence of :class:`Wordlist` (lists, tuples); caches on the
    hashable ``(tuple(wordlists), max_n)`` so identical vocabulary reuses one
    frozen config across runs (replaces the legacy
    ``normalize_wordlists_for_cache`` + ``@st.cache_data`` dance).

    Category names are taken as-is from each wordlist and are expected to be
    **unique across all wordlists** (``core.io.merge_wordlists`` namespaces
    them via :func:`core.naming.sanitize_identifier`). No namespacing is done
    here — this module is pure over the configured vocabulary.
    """
    return _build_analysis_config_cached(tuple(wordlists), max_n)


# --------------------------------------------------------------------------- #
# Per-document counting (PURE)
# --------------------------------------------------------------------------- #
def _accumulate(
    term_counter: Counter[str],
    exact_lookup: Mapping[str, tuple[str, ...]],
    wildcard_trie: Trie,
    category_counts: dict[str, int],
    category_detected: dict[str, dict[str, int]],
) -> None:
    """Apply the single token-frequency convention for one (exact, wildcard) pair.

    Exact and wildcard branches both add **occurrences** (fixes §2.1) and both
    record ``{term: occ}`` in ``category_detected`` (fixes §2.2 at the source).
    Used once for unigrams and once for n-grams.
    """
    for term, occ in term_counter.items():
        for cat in exact_lookup.get(term, ()):
            category_counts[cat] += occ
            category_detected[cat][term] = occ
        if wildcard_trie:
            for cat in _match_prefix_categories(term, wildcard_trie):
                category_counts[cat] += occ
                category_detected[cat][term] = occ


def count_document(document: object, config: AnalysisConfig) -> DocumentCounts:
    """Count one document against a prebuilt config.

    ``document`` is normally a string; non-strings (``None``, ``NaN``, numbers)
    are treated as empty so the pure function is robust standalone — callers
    need not pre-coerce. Returns a frozen :class:`DocumentCounts` whose
    ``category_counts`` and ``category_detected`` use the single token-frequency
    convention for exact *and* wildcard, unigrams *and* n-grams (fixes §2.1).
    """
    if not isinstance(document, str):
        document = ""

    tokens = tokenize(document)
    n_tokens = len(tokens)
    n_types = len(set(tokens))

    category_counts: dict[str, int] = {c: 0 for c in config.categories}
    category_detected: dict[str, dict[str, int]] = {c: {} for c in config.categories}

    if tokens:
        _accumulate(
            Counter(tokens),
            config.exact_single_lookup,
            config.wildcard_single_trie,
            category_counts,
            category_detected,
        )

    # N-grams (multi-word phrases).
    if config.required_ngram_lengths and n_tokens >= MIN_NGRAM_LENGTH:
        _accumulate(
            generate_ngrams(tokens, config.required_ngram_lengths),
            config.exact_multi_lookup,
            config.wildcard_multi_trie,
            category_counts,
            category_detected,
        )

    return DocumentCounts(
        n_tokens=n_tokens,
        n_types=n_types,
        category_counts=MappingProxyType(category_counts),
        category_detected=MappingProxyType(
            {c: MappingProxyType(d) for c, d in category_detected.items()}
        ),
    )


# --------------------------------------------------------------------------- #
# Batch analysis → DataFrame
# --------------------------------------------------------------------------- #
def analyze_documents(
    documents: Sequence[str],
    config: AnalysisConfig,
    *,
    progress: Callable[[int, int], None] | None = None,
    executor: Executor | None = None,
) -> pd.DataFrame:
    """Analyze every document, returning the analysis DataFrame.

    Columns: ``n_tokens``, ``n_types``, and for each category ``{cat}_word_count``,
    ``{cat}_word_perc``, ``{cat}_detected_words`` (the last a ``dict[str,int]``
    per row — frequency travels in the struct, fixes §2.2).

    ``progress(completed, total)`` is called after each document completes
    (§5.4 — the only progress surface; UI/CLI/API feed different closures).
    ``executor`` (any ``concurrent.futures.Executor``) parallelizes the
    per-document work (§5.5); ``None`` runs serially. With an executor,
    ``executor.map`` preserves submission order, so ``completed`` is monotonic.
    """
    docs = list(documents)
    total = len(docs)
    results: list[DocumentCounts | None] = [None] * total

    completed = 0

    def _work(index: int) -> tuple[int, DocumentCounts]:
        return index, count_document(docs[index], config)

    if executor is not None:
        for index, counts in executor.map(_work, range(total)):
            results[index] = counts
            completed += 1
            if progress is not None:
                progress(completed, total)
    else:
        for index in range(total):
            results[index] = count_document(docs[index], config)
            completed += 1
            if progress is not None:
                progress(completed, total)

    filled: list[DocumentCounts] = [
        r if r is not None else DocumentCounts.empty(config.categories) for r in results
    ]

    columns: dict[str, list[Any]] = {
        "n_tokens": [r.n_tokens for r in filled],
        "n_types": [r.n_types for r in filled],
    }
    for cat in config.categories:
        columns[f"{cat}_word_count"] = [r.category_counts[cat] for r in filled]
        columns[f"{cat}_word_perc"] = [
            (r.category_counts[cat] / r.n_tokens) if r.n_tokens > 0 else 0.0 for r in filled
        ]
        columns[f"{cat}_detected_words"] = [dict(r.category_detected[cat]) for r in filled]

    return pd.DataFrame(columns)


__all__ = [
    "build_analysis_config",
    "count_document",
    "analyze_documents",
]
