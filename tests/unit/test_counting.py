"""Regression suite for ``wordcount.core.counting`` — the §2.1 fix and the
composability invariants the audit said were missing.

The headline invariant is :func:`test_exact_equals_wildcard`: the same document
counts the same whether a term is written exactly (``happy``) or as a wildcard
(``happ*``). Before the fix, exact counted *types* (1) while wildcard counted
*tokens* (3) — silently incomparable metrics (audit §2.1, critical).

Pure core tests: pandas is used only to inspect the analysis DataFrame; no
streamlit, no fastapi.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import cast

import pandas as pd
import pytest

from wordcount.core.counting import (
    _build_prefix_trie,
    _match_prefix_categories,
    analyze_documents,
    build_analysis_config,
    count_document,
)
from wordcount.core.models import AnalysisConfig, CategoryTerms, Wordlist


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _wordlist(
    namespace: str = "wl",
    *,
    exact_single: frozenset[str] | set[str] = frozenset(),
    wildcard_single: tuple[str, ...] | list[str] = (),
    exact_multi: frozenset[str] | set[str] = frozenset(),
    wildcard_multi: tuple[str, ...] | list[str] = (),
    category: str = "Affect",
) -> Wordlist:
    """One wordlist with a single category carrying the given terms."""
    terms = CategoryTerms(
        exact_single=frozenset(exact_single),
        wildcard_single=tuple(wildcard_single),
        exact_multi=frozenset(exact_multi),
        wildcard_multi=tuple(wildcard_multi),
    )
    return Wordlist.from_mapping(namespace, {category: terms})


def _config(wordlists: list[Wordlist], max_n: int = 3) -> AnalysisConfig:
    return build_analysis_config(wordlists, max_n=max_n)


# --------------------------------------------------------------------------- #
# §2.1 — exact counts TOKENS not types (the critical fix)
# --------------------------------------------------------------------------- #
def test_exact_counts_tokens_not_types() -> None:
    wl = _wordlist(exact_single={"happy"})
    config = _config([wl])
    counts = count_document("happy happy happy", config)
    # Was 1 (types); now 3 (token occurrences).
    assert counts.category_counts["Affect"] == 3
    assert counts.n_tokens == 3


def test_wildcard_counts_tokens() -> None:
    wl = _wordlist(wildcard_single=("happ",))
    config = _config([wl])
    counts = count_document("happy happy happy", config)
    assert counts.category_counts["Affect"] == 3


def test_exact_equals_wildcard_unigram() -> None:
    """The headline invariant: same count for exact vs wildcard, same doc."""
    doc = "happy happy happy sad"
    exact = count_document(doc, _config([_wordlist(exact_single={"happy"})]))
    wild = count_document(doc, _config([_wordlist(wildcard_single=("happ",))]))
    assert exact.category_counts["Affect"] == wild.category_counts["Affect"] == 3


def test_exact_counts_single_occurrence() -> None:
    wl = _wordlist(exact_single={"happy"})
    config = _config([wl])
    counts = count_document("a happy day", config)
    assert counts.category_counts["Affect"] == 1


# --------------------------------------------------------------------------- #
# §2.1 — n-gram variants (exact + wildcard both count occurrences)
# --------------------------------------------------------------------------- #
def test_exact_ngram_counts_tokens() -> None:
    wl = _wordlist(exact_multi={"new york"}, category="Place")
    config = _config([wl], max_n=3)
    # "new york" appears twice.
    counts = count_document("new york new york baby", config)
    assert counts.category_counts["Place"] == 2


def test_wildcard_ngram_counts_tokens() -> None:
    wl = _wordlist(wildcard_multi=("new",), category="Place")
    config = _config([wl], max_n=2)
    counts = count_document("new york new york baby", config)
    # With only bigrams generated, "new" matches the two "new york" bigrams.
    # Occurrences are counted (types would be 1), not deduped.
    assert counts.category_counts["Place"] == 2


def test_wildcard_multi_matches_across_orders() -> None:
    """A 1-word multi-wildcard prefix matches n-grams of every generated order.

    This is the preserved ``ace8366`` algorithm: prefix "new" (from "new *")
    forces orders 2..max_n, and the trie matches any joined phrase starting
    with "new". With max_n=3 on "new york new haven baby" it hits the two
    bigrams AND the two trigrams => 4. Locks the behavior in.
    """
    wl = _wordlist(wildcard_multi=("new",), category="Place")
    config = _config([wl], max_n=3)
    counts = count_document("new york new haven baby", config)
    assert counts.category_counts["Place"] == 4


def test_exact_equals_wildcard_ngram() -> None:
    # max_n=2 so the wildcard's scope (any bigram starting with "new y") equals
    # the exact phrase's order; with higher max_n the wildcard would also catch
    # longer phrases, a separate (intentional, preserved) scope difference.
    doc = "new york new york baby"
    exact = _config([_wordlist(exact_multi={"new york"}, category="Place")], max_n=2)
    wild = _config([_wordlist(wildcard_multi=("new y",), category="Place")], max_n=2)
    assert (
        count_document(doc, exact).category_counts["Place"]
        == count_document(doc, wild).category_counts["Place"]
        == 2
    )


# --------------------------------------------------------------------------- #
# Multiple categories / detected words carry frequency (§2.2 at the source)
# --------------------------------------------------------------------------- #
def test_term_in_multiple_categories() -> None:
    wl = Wordlist.from_mapping(
        "wl",
        {
            "Affect": CategoryTerms(
                exact_single=frozenset({"happy"}),
                wildcard_single=(),
                exact_multi=frozenset(),
                wildcard_multi=(),
            ),
            "Positive": CategoryTerms(
                exact_single=frozenset({"happy"}),
                wildcard_single=(),
                exact_multi=frozenset(),
                wildcard_multi=(),
            ),
        },
    )
    config = _config([wl])
    counts = count_document("happy happy", config)
    assert counts.category_counts["Affect"] == 2
    assert counts.category_counts["Positive"] == 2


def test_detected_words_carry_frequency() -> None:
    wl = _wordlist(exact_single={"happy", "sad"})
    config = _config([wl])
    counts = count_document("happy happy sad", config)
    detected = dict(counts.category_detected["Affect"])
    assert detected == {"happy": 2, "sad": 1}


def test_wildcard_detected_words_carry_frequency() -> None:
    wl = _wordlist(wildcard_single=("happ",))
    config = _config([wl])
    counts = count_document("happy happy happening", config)
    detected = dict(counts.category_detected["Affect"])
    assert detected == {"happy": 2, "happening": 1}


# --------------------------------------------------------------------------- #
# Empty / NaN / non-string documents
# --------------------------------------------------------------------------- #
def test_empty_document() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    counts = count_document("", config)
    assert counts.n_tokens == 0
    assert counts.n_types == 0
    assert counts.category_counts["Affect"] == 0
    assert dict(counts.category_detected["Affect"]) == {}


def test_nan_document_is_treated_as_empty() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    nan = cast("str", float("nan"))
    counts = count_document(nan, config)
    assert counts.n_tokens == 0
    assert counts.category_counts["Affect"] == 0


def test_none_document_is_treated_as_empty() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    counts = count_document(None, config)  # type: ignore[arg-type]
    assert counts.n_tokens == 0


def test_document_with_no_matches() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    counts = count_document("nothing here matches", config)
    assert counts.n_tokens == 3
    assert counts.category_counts["Affect"] == 0


# --------------------------------------------------------------------------- #
# §5.4 — progress callback
# --------------------------------------------------------------------------- #
def test_progress_callback_invoked() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    calls: list[tuple[int, int]] = []
    analyze_documents(
        ["happy", "happy happy", "none", ""],
        config,
        progress=lambda done, total: calls.append((done, total)),
    )
    assert len(calls) == 4
    assert calls[0] == (1, 4)
    assert calls[-1] == (4, 4)


def test_progress_callback_not_required() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    df = analyze_documents(["happy", "none"], config)
    assert len(df) == 2


# --------------------------------------------------------------------------- #
# §5.5 — parallelism matches serial
# --------------------------------------------------------------------------- #
def test_parallel_matches_serial() -> None:
    wl = _wordlist(
        exact_single={"happy", "sad"},
        wildcard_single=("happ",),
        exact_multi={"new york"},
        wildcard_multi=("new",),
        category="Mixed",
    )
    config = _config([wl], max_n=3)
    docs = [
        "happy happy sad",
        "new york new york happy",
        "nothing here at all",
        "happening happily",
        "",
        "new haven new jersey",
    ]

    serial = analyze_documents(docs, config)
    with ThreadPoolExecutor(max_workers=3) as ex:
        parallel = analyze_documents(docs, config, executor=ex)

    pd.testing.assert_frame_equal(serial, parallel)


def test_parallel_progress_monotonic() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    docs = ["happy"] * 10
    calls: list[tuple[int, int]] = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        analyze_documents(docs, config, executor=ex, progress=lambda d, t: calls.append((d, t)))
    assert calls[-1] == (10, 10)
    assert [d for d, _ in calls] == sorted(d for d, _ in calls)


# --------------------------------------------------------------------------- #
# Caching — build_analysis_config reuses one config for equal vocabulary
# (replaces the plan's test_config_is_hashable: AnalysisConfig is intentionally
#  NOT hashable because of its trie; the cache key is the hashable Wordlists.)
# --------------------------------------------------------------------------- #
def test_build_analysis_config_caches_equal_vocabulary() -> None:
    wl1 = _wordlist(exact_single={"happy"})
    wl2 = _wordlist(exact_single={"happy"})  # equal but distinct instance
    assert wl1 == wl2
    assert hash(wl1) == hash(wl2)

    c1 = build_analysis_config([wl1], max_n=3)
    c2 = build_analysis_config([wl2], max_n=3)
    # lru_cache hit: same object.
    assert c1 is c2


def test_build_analysis_config_different_vocabulary_distinct() -> None:
    c1 = build_analysis_config([_wordlist(exact_single={"happy"})], max_n=3)
    c2 = build_analysis_config([_wordlist(exact_single={"sad"})], max_n=3)
    assert c1 is not c2
    assert c1.categories == c2.categories  # both "Affect"
    assert c1.exact_single_lookup != c2.exact_single_lookup


def test_wordlist_is_hashable_and_equal() -> None:
    a = _wordlist(exact_single={"happy", "sad"}, wildcard_single=("h",))
    b = _wordlist(exact_single={"sad", "happy"}, wildcard_single=("h",))
    # CategoryTerms sorts internally, so construction order doesn't matter.
    assert a == b
    assert hash(a) == hash(b)
    # Usable as a dict key (the lru_cache requirement).
    d: dict[Wordlist, int] = {a: 1}
    assert d[b] == 1


# --------------------------------------------------------------------------- #
# analyze_documents DataFrame shape & derived columns
# --------------------------------------------------------------------------- #
def test_analyze_documents_columns() -> None:
    wl = _wordlist(exact_single={"happy"}, category="Affect")
    config = _config([wl])
    df = analyze_documents(["happy happy", "none"], config)
    assert list(df.columns) == [
        "n_tokens",
        "n_types",
        "Affect_word_count",
        "Affect_word_perc",
        "Affect_detected_words",
    ]


def test_analyze_documents_word_perc() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    df = analyze_documents(["happy happy sad"], config)
    assert df.loc[0, "Affect_word_count"] == 2
    assert df.loc[0, "n_tokens"] == 3
    assert df.loc[0, "Affect_word_perc"] == pytest.approx(2 / 3)


def test_analyze_documents_word_perc_zero_for_empty() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    df = analyze_documents(["", "none"], config)
    assert df.loc[0, "Affect_word_perc"] == 0.0
    assert df.loc[1, "Affect_word_perc"] == 0.0


def test_analyze_documents_detected_words_are_dicts_with_frequency() -> None:
    config = _config([_wordlist(exact_single={"happy"})])
    df = analyze_documents(["happy happy"], config)
    assert df.loc[0, "Affect_detected_words"] == {"happy": 2}


# --------------------------------------------------------------------------- #
# max_n honored end-to-end (§5.1)
# --------------------------------------------------------------------------- #
def test_max_n_controls_ngram_orders() -> None:
    wl = _wordlist(exact_multi={"a b c"}, category="Phrase")
    # With max_n=2 the 3-gram "a b c" is never generated -> 0.
    cfg_2 = _config([wl], max_n=2)
    assert count_document("a b c a b c", cfg_2).category_counts["Phrase"] == 0
    # With max_n=3 it matches twice.
    cfg_3 = _config([wl], max_n=3)
    assert count_document("a b c a b c", cfg_3).category_counts["Phrase"] == 2


# --------------------------------------------------------------------------- #
# Ported trie helpers + defensive guards
# --------------------------------------------------------------------------- #
def test_match_prefix_categories_empty_trie_returns_empty() -> None:
    assert _match_prefix_categories("anything", {}) == []


def test_match_prefix_categories_walks_terminal_nodes() -> None:
    trie = _build_prefix_trie({"ha": ["Affect"], "hap": ["Affect", "Mood"]})
    # "happy" walks h-a (terminal Affect) then h-a-p (terminal Affect, Mood).
    assert _match_prefix_categories("happy", trie) == ["Affect", "Affect", "Mood"]
    # "nope" shares no prefix.
    assert _match_prefix_categories("nope", trie) == []


def test_empty_wildcard_prefix_is_skipped() -> None:
    # An empty-string multi prefix must be skipped (not crash, not force n-grams).
    wl = _wordlist(wildcard_multi=("",), category="Place")
    config = _config([wl], max_n=3)
    assert config.required_ngram_lengths == ()
    assert config.wildcard_multi_trie == {}
    counts = count_document("new york new york", config)
    assert counts.category_counts["Place"] == 0
