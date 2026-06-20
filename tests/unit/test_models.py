"""Tests for the interchange dataclasses (core/models.py).

These lock in the contract that later phases depend on:
* CategoryTerms / Wordlist are hashable (cache keys).
* DocumentCounts / AnalysisConfig / stats results are frozen but NOT hashable.
* category_detected carries term->occurrences frequency.
* Errors form a single catchable hierarchy with stable codes.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from wordcount.core.models import (
    AnalysisConfig,
    AnalysisConfigError,
    AnovaResult,
    CategoryTerms,
    DocumentCounts,
    GroupStats,
    MissingDicTermColumnError,
    NoCategoryColumnsError,
    NoTextColumnError,
    PearsonResult,
    UnsupportedFormatError,
    WordcountError,
    Wordlist,
)


# --------------------------------------------------------------------------- #
# CategoryTerms
# --------------------------------------------------------------------------- #
def test_category_terms_is_frozen_and_hashable() -> None:
    terms = CategoryTerms(
        exact_single=frozenset({"happy"}),
        wildcard_single=("happ",),
        exact_multi=frozenset({"very good"}),
        wildcard_multi=("very g",),
    )
    # frozen: cannot reassign
    with pytest.raises(FrozenInstanceError):
        terms.exact_single = frozenset({"sad"})  # type: ignore[misc]
    # hashable: usable as dict key / set member
    assert hash(terms) == hash(terms)
    assert terms in {terms}


def test_category_terms_equality_is_value_based() -> None:
    a = CategoryTerms(frozenset({"a"}), ("b",), frozenset(), ())
    b = CategoryTerms(frozenset({"a"}), ("b",), frozenset(), ())
    assert a == b
    assert hash(a) == hash(b)


def test_category_terms_empty_and_count() -> None:
    empty = CategoryTerms.empty()
    assert empty.n_terms == 0
    terms = CategoryTerms(
        exact_single=frozenset({"a", "b"}),
        wildcard_single=("c",),
        exact_multi=frozenset({"d e"}),
        wildcard_multi=(),
    )
    assert terms.n_terms == 4


# --------------------------------------------------------------------------- #
# Wordlist (hashable cache key)
# --------------------------------------------------------------------------- #
def _make_wordlist(namespace: str = "wl") -> Wordlist:
    return Wordlist.from_mapping(
        namespace,
        {
            "Affect": CategoryTerms(
                exact_single=frozenset({"happy", "sad"}),
                wildcard_single=("happ",),
                exact_multi=frozenset(),
                wildcard_multi=(),
            ),
            "Modal": CategoryTerms(
                exact_single=frozenset({"can"}),
                wildcard_single=(),
                exact_multi=frozenset(),
                wildcard_multi=(),
            ),
        },
    )


def test_wordlist_is_hashable_and_order_independent() -> None:
    # Built from a dict in different insertion orders -> same hash/equality
    # because from_mapping sorts.
    wl_a = _make_wordlist("wl")
    wl_b = Wordlist.from_mapping(
        "wl",
        {
            "Modal": CategoryTerms(
                exact_single=frozenset({"can"}),
                wildcard_single=(),
                exact_multi=frozenset(),
                wildcard_multi=(),
            ),
            "Affect": CategoryTerms(
                exact_single=frozenset({"happy", "sad"}),
                wildcard_single=("happ",),
                exact_multi=frozenset(),
                wildcard_multi=(),
            ),
        },
    )
    assert wl_a == wl_b
    assert hash(wl_a) == hash(wl_b)
    assert wl_a in {wl_b}


def test_wordlist_accessors() -> None:
    wl = _make_wordlist()
    assert set(wl.category_names) == {"Affect", "Modal"}
    assert wl.n_categories == 2
    assert "happy" in wl.terms_for("Affect").exact_single
    assert wl.terms_for("Modal").exact_single == frozenset({"can"})


def test_wordlist_terms_for_missing_raises() -> None:
    wl = _make_wordlist()
    with pytest.raises(KeyError):
        wl.terms_for("Nonexistent")


def test_wordlist_is_frozen() -> None:
    wl = _make_wordlist()
    with pytest.raises(FrozenInstanceError):
        wl.namespace = "other"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# AnalysisConfig — frozen but NOT hashable (trie)
# --------------------------------------------------------------------------- #
def test_analysis_config_is_frozen_but_not_hashable() -> None:
    config = AnalysisConfig(
        categories=("Affect",),
        exact_single_lookup={"happy": ("Affect",)},
        exact_multi_lookup={},
        wildcard_single_trie={"h": {"a": {"p": {"p": {"_categories_": ["Affect"]}}}}},
        wildcard_multi_trie={},
        required_ngram_lengths=(2,),
        max_n=3,
    )
    assert config.has_wildcards
    with pytest.raises(FrozenInstanceError):
        config.max_n = 5  # type: ignore[misc]
    # The trie (nested dict) is unhashable -> the dataclass is unhashable.
    with pytest.raises(TypeError):
        hash(config)


def test_analysis_config_has_wildcards_false_when_empty() -> None:
    config = AnalysisConfig(
        categories=(),
        exact_single_lookup={},
        exact_multi_lookup={},
        wildcard_single_trie={},
        wildcard_multi_trie={},
        required_ngram_lengths=(),
        max_n=3,
    )
    assert not config.has_wildcards


# --------------------------------------------------------------------------- #
# DocumentCounts — frequency travels in the struct
# --------------------------------------------------------------------------- #
def test_document_counts_carries_frequency() -> None:
    counts = DocumentCounts(
        n_tokens=3,
        n_types=1,
        category_counts={"Affect": 3},
        category_detected={"Affect": {"happy": 3}},
    )
    # The §2.2 fix: term -> occurrences, no string join.
    assert counts.category_detected["Affect"]["happy"] == 3
    # frozen
    with pytest.raises(FrozenInstanceError):
        counts.n_tokens = 0  # type: ignore[misc]


def test_document_counts_empty_factory() -> None:
    counts = DocumentCounts.empty(("Affect", "Modal"))
    assert counts.n_tokens == 0
    assert counts.category_counts["Affect"] == 0
    assert dict(counts.category_detected["Modal"]) == {}


# --------------------------------------------------------------------------- #
# Stats results — frozen
# --------------------------------------------------------------------------- #
def test_pearson_result_is_frozen() -> None:
    r = PearsonResult(col1="a", col2="b", coefficient=0.5, p_value=0.01, n=42)
    with pytest.raises(FrozenInstanceError):
        r.coefficient = 0.0  # type: ignore[misc]
    assert r.n == 42


def test_group_stats_is_frozen() -> None:
    g = GroupStats(category="A", n=10, mean=1.0, sem=0.1, ci_lower=0.8, ci_upper=1.2)
    with pytest.raises(FrozenInstanceError):
        g.mean = 2.0  # type: ignore[misc]
    assert g.ci_lower < g.mean < g.ci_upper


def test_anova_result_is_frozen() -> None:
    table = pd.DataFrame({"PR(>F)": [0.03]}, index=["C(Affect)"])
    result = AnovaResult(
        cat_var="Affect",
        num_var="score",
        p_value=0.03,
        significant=True,
        table=table,
        group_stats=(GroupStats("A", 5, 1.0, 0.1, 0.8, 1.2),),
        tukey_rows=({"group1": "A", "group2": "B", "reject": True},),
    )
    with pytest.raises(FrozenInstanceError):
        result.significant = False  # type: ignore[misc]
    assert result.significant is True
    assert len(result.group_stats) == 1


# --------------------------------------------------------------------------- #
# Error hierarchy
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("exc_cls", "expected_code"),
    [
        (UnsupportedFormatError, "unsupported_format"),
        (MissingDicTermColumnError, "missing_dicterm_column"),
        (NoCategoryColumnsError, "no_category_columns"),
        (NoTextColumnError, "no_text_column"),
        (AnalysisConfigError, "analysis_config_error"),
    ],
)
def test_errors_have_stable_codes_and_share_base(
    exc_cls: type[WordcountError], expected_code: str
) -> None:
    exc = exc_cls("boom")
    assert isinstance(exc, WordcountError)
    assert exc.code == expected_code
    # catchable as the base
    try:
        raise exc_cls("boom")
    except WordcountError as caught:
        assert caught.code == expected_code


def test_error_code_overridable() -> None:
    exc = WordcountError("x", code="custom")
    assert exc.code == "custom"
