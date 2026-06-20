"""Tests for ``wordcount.viz.plots`` — the §2.2 fix (read structured
``{term: occ}`` dicts, no string re-split) and the long-form DataFrame shape.
"""

from __future__ import annotations

import pandas as pd

from wordcount.viz.plots import aggregate_detected, build_barplot_data


# --------------------------------------------------------------------------- #
# Structured dict input (the canonical form from core/counting.py)
# --------------------------------------------------------------------------- #
def test_build_barplot_data_reads_structured_dicts() -> None:
    series = pd.Series(
        [
            {"happy": 2, "sad": 1},
            {"happy": 1, "angry": 3},
            {},
        ]
    )
    df = build_barplot_data(series, top_n=10)
    assert list(df.columns) == ["term", "frequency", "doc_frequency"]
    # happy: 2+1=3 occurrences across 2 docs; angry: 3 across 1; sad: 1 across 1.
    by_term = df.set_index("term")
    assert by_term.loc["happy", "frequency"] == 3
    assert by_term.loc["happy", "doc_frequency"] == 2
    assert by_term.loc["angry", "frequency"] == 3
    assert by_term.loc["angry", "doc_frequency"] == 1
    assert by_term.loc["sad", "frequency"] == 1


def test_build_barplot_data_sorted_by_frequency_desc() -> None:
    series = pd.Series([{"rare": 1, "common": 5, "mid": 3}])
    df = build_barplot_data(series, top_n=10)
    assert list(df["term"]) == ["common", "mid", "rare"]


def test_build_barplot_data_top_n_limit() -> None:
    series = pd.Series([{"a": 1, "b": 2, "c": 3, "d": 4}])
    df = build_barplot_data(series, top_n=2)
    assert len(df) == 2
    assert list(df["term"]) == ["d", "c"]


def test_build_barplot_data_tiebreak_alphabetical() -> None:
    # Equal frequencies -> alphabetical tiebreak for determinism.
    series = pd.Series([{"zebra": 2, "apple": 2, "mango": 2}])
    df = build_barplot_data(series, top_n=10)
    assert list(df["term"]) == ["apple", "mango", "zebra"]


def test_build_barplot_data_empty_series() -> None:
    df = build_barplot_data(pd.Series([], dtype=object), top_n=5)
    assert df.empty
    assert list(df.columns) == ["term", "frequency", "doc_frequency"]


def test_build_barplot_data_all_empty_dicts() -> None:
    series = pd.Series([{}, {}, {}])
    df = build_barplot_data(series, top_n=5)
    assert df.empty


def test_build_barplot_data_skips_nan_cells() -> None:
    series = pd.Series([{"happy": 1}, None, float("nan")])
    df = build_barplot_data(series, top_n=5)
    assert list(df["term"]) == ["happy"]
    assert df.loc[0, "frequency"] == 1


# --------------------------------------------------------------------------- #
# §2.2 regression — frequency from structured dicts matches occurrences
# --------------------------------------------------------------------------- #
def test_frequency_is_token_occurrences_not_doc_count() -> None:
    """The headline §2.2 invariant: frequency = total occurrences, not doc count.

    Before, the legacy re-split string + Counter gave doc-frequency-ish numbers;
    now the structured {term: occ} dict carries occurrence counts directly.
    """
    series = pd.Series(
        [
            {"happy": 5},  # 5 occurrences in 1 doc
            {"happy": 1},  # 1 occurrence in another doc
        ]
    )
    df = build_barplot_data(series, top_n=5)
    assert df.loc[0, "term"] == "happy"
    assert df.loc[0, "frequency"] == 6  # 5 + 1
    assert df.loc[0, "doc_frequency"] == 2  # 2 docs


# --------------------------------------------------------------------------- #
# String fallback (tolerant path for already-enhanced frames)
# --------------------------------------------------------------------------- #
def test_build_barplot_data_string_fallback() -> None:
    # Enhanced frames serialize dicts to "term: cnt, ..." strings.
    series = pd.Series(["happy: 2, sad: 1", "happy: 1", ""])
    df = build_barplot_data(series, top_n=10)
    by_term = df.set_index("term")
    assert by_term.loc["happy", "frequency"] == 3
    assert by_term.loc["sad", "frequency"] == 1


def test_build_barplot_data_legacy_list_string_form() -> None:
    # The legacy ", "-joined form (no counts) is tolerated; each term counts 1.
    series = pd.Series(["happy, sad, happy"])
    df = build_barplot_data(series, top_n=10)
    by_term = df.set_index("term")
    assert by_term.loc["happy", "frequency"] == 2
    assert by_term.loc["sad", "frequency"] == 1


def test_build_barplot_data_string_fallback_skips_bad_counts() -> None:
    # A non-numeric count after ": " is skipped (ValueError path).
    series = pd.Series(["happy: 2, sad: notanum"])
    df = build_barplot_data(series, top_n=10)
    assert list(df["term"]) == ["happy"]
    assert df.loc[0, "frequency"] == 2


def test_build_barplot_data_string_fallback_skips_empty_pairs() -> None:
    # A string with stray ", " separators yields empty pairs -> skipped.
    series = pd.Series(["happy: 2, , sad: 1"])
    df = build_barplot_data(series, top_n=10)
    assert set(df["term"]) == {"happy", "sad"}


def test_aggregate_detected_via_cells_sequence() -> None:
    terms, freqs, doc_freqs = aggregate_detected(
        pd.Series([], dtype=object),
        cells=[{"happy": 2}, {"happy": 1, "sad": 4}],
    )
    # sad (4) > happy (3) -> sad first by descending frequency.
    assert terms == ["sad", "happy"]
    assert freqs == [4, 3]
    assert doc_freqs == [1, 2]
