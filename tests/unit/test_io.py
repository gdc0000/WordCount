"""Tests for ``wordcount.core.io`` — pure loaders + typed exceptions.

Covers each supported format (csv/tsv/xls/xlsx for datasets; csv/tsv/txt/dic/
dicx/xls/xlsx for wordlists), the typed exceptions
(``UnsupportedFormatError`` / ``MissingDicTermColumnError`` /
``NoCategoryColumnsError``), ``dtype``/``usecols`` forwarding (§5.7), and
``merge_wordlists`` namespacing + global uniqueness (§10 scheme preserved).

Pure core tests: pandas is used to build fixtures; no streamlit, no fastapi.
"""

from __future__ import annotations

import io as _io
from pathlib import Path

import pandas as pd
import pytest

from wordcount.core.counting import build_analysis_config, count_document
from wordcount.core.io import merge_wordlists, read_dataset, read_wordlist
from wordcount.core.models import (
    MissingDicTermColumnError,
    NoCategoryColumnsError,
    UnsupportedFormatError,
    Wordlist,
)

# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
DATASET_DF = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "text": ["happy day", "sad sad night", "new york baby"],
        "score": [0.1, 0.2, 0.3],
    }
)

WORDLIST_DF = pd.DataFrame(
    {
        "DicTerm": ["happy", "sad", "happ*", "new york", "new *"],
        "Affect": ["X", "X", "X", "", ""],
        "Place": ["", "", "", "X", "X"],
    }
)


def _write_csv(path: Path, df: pd.DataFrame, sep: str = ",") -> Path:
    df.to_csv(path, sep=sep, index=False)
    return path


def _write_xlsx(path: Path, df: pd.DataFrame) -> Path:
    df.to_excel(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# read_dataset — formats
# --------------------------------------------------------------------------- #
def test_read_dataset_csv(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "data.csv", DATASET_DF)
    df = read_dataset(path)
    assert list(df.columns) == ["id", "text", "score"]
    assert len(df) == 3


def test_read_dataset_tsv(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "data.tsv", DATASET_DF, sep="\t")
    df = read_dataset(path)
    assert list(df.columns) == ["id", "text", "score"]
    assert df.loc[0, "text"] == "happy day"


def test_read_dataset_xlsx(tmp_path: Path) -> None:
    path = _write_xlsx(tmp_path / "data.xlsx", DATASET_DF)
    df = read_dataset(path)
    assert list(df.columns) == ["id", "text", "score"]
    assert len(df) == 3


def test_read_dataset_xlsx_dtype_and_usecols(tmp_path: Path) -> None:
    path = _write_xlsx(tmp_path / "data.xlsx", DATASET_DF)
    df = read_dataset(path, dtype={"score": "float32"}, usecols=["text", "score"])
    assert list(df.columns) == ["text", "score"]
    assert str(df["score"].dtype) == "float32"


def test_read_dataset_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text("[]")
    with pytest.raises(UnsupportedFormatError) as exc:
        read_dataset(path)
    assert exc.value.code == "unsupported_format"


def test_read_dataset_unsupported_suffix_in_message(tmp_path: Path) -> None:
    path = tmp_path / "weird.parquet"
    path.write_bytes(b"")
    with pytest.raises(UnsupportedFormatError, match="parquet"):
        read_dataset(path)


# --------------------------------------------------------------------------- #
# read_dataset — dtype / usecols (§5.7)
# --------------------------------------------------------------------------- #
def test_read_dataset_dtype_forwarded(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "data.csv", DATASET_DF)
    df = read_dataset(path, dtype={"id": "str", "score": "float32"})
    assert df["id"].dtype == object  # str -> object
    assert str(df["score"].dtype) == "float32"


def test_read_dataset_usecols_forwarded(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "data.csv", DATASET_DF)
    df = read_dataset(path, usecols=["id", "text"])
    assert list(df.columns) == ["id", "text"]


# --------------------------------------------------------------------------- #
# read_dataset — file-like sources
# --------------------------------------------------------------------------- #
def test_read_dataset_from_filelike() -> None:
    buf = _io.StringIO()
    DATASET_DF.to_csv(buf, index=False)
    buf.seek(0)
    buf.name = "data.csv"  # type: ignore[attr-defined]
    df = read_dataset(buf)
    assert list(df.columns) == ["id", "text", "score"]


def test_read_dataset_filelike_with_explicit_name() -> None:
    buf = _io.StringIO()
    DATASET_DF.to_csv(buf, index=False, sep="\t")  # write TSV content
    buf.seek(0)
    df = read_dataset(buf, name="data.tsv")
    assert list(df.columns) == ["id", "text", "score"]


# --------------------------------------------------------------------------- #
# read_wordlist — formats
# --------------------------------------------------------------------------- #
def test_read_wordlist_csv(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "wl.csv", WORDLIST_DF)
    result = read_wordlist(path)
    assert set(result) == {"Affect", "Place"}
    affect = result["Affect"]
    assert "happy" in affect.exact_single
    assert "sad" in affect.exact_single
    assert "happ" in affect.wildcard_single  # * stripped
    assert affect.wildcard_multi == ()
    place = result["Place"]
    assert "new york" in place.exact_multi
    assert "new" in place.wildcard_multi  # "new *" -> "new"


def test_read_wordlist_tsv(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "wl.tsv", WORDLIST_DF, sep="\t")
    result = read_wordlist(path)
    assert "happ" in result["Affect"].wildcard_single


def test_read_wordlist_dic(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "wl.dic", WORDLIST_DF, sep="\t")
    result = read_wordlist(path)
    assert set(result) == {"Affect", "Place"}


def test_read_wordlist_txt(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "wl.txt", WORDLIST_DF, sep="\t")
    result = read_wordlist(path)
    assert set(result) == {"Affect", "Place"}


def test_read_wordlist_xlsx(tmp_path: Path) -> None:
    path = _write_xlsx(tmp_path / "wl.xlsx", WORDLIST_DF)
    result = read_wordlist(path)
    assert "happy" in result["Affect"].exact_single
    assert "new york" in result["Place"].exact_multi


def test_read_wordlist_dicx(tmp_path: Path) -> None:
    # dicx is read as comma-separated (same engine as csv).
    path = _write_csv(tmp_path / "wl.dicx", WORDLIST_DF, sep=",")
    result = read_wordlist(path)
    assert set(result) == {"Affect", "Place"}


def test_read_wordlist_unsupported_format(tmp_path: Path) -> None:
    path = tmp_path / "wl.json"
    path.write_text("{}")
    with pytest.raises(UnsupportedFormatError) as exc:
        read_wordlist(path)
    assert exc.value.code == "unsupported_format"


# --------------------------------------------------------------------------- #
# read_wordlist — typed exceptions
# --------------------------------------------------------------------------- #
def test_read_wordlist_missing_dicterm(tmp_path: Path) -> None:
    bad = pd.DataFrame({"Term": ["happy"], "Affect": ["X"]})
    path = _write_csv(tmp_path / "wl.csv", bad)
    with pytest.raises(MissingDicTermColumnError) as exc:
        read_wordlist(path)
    assert exc.value.code == "missing_dicterm_column"


def test_read_wordlist_no_category_columns(tmp_path: Path) -> None:
    bad = pd.DataFrame({"DicTerm": ["happy", "sad"]})
    path = _write_csv(tmp_path / "wl.csv", bad)
    with pytest.raises(NoCategoryColumnsError) as exc:
        read_wordlist(path)
    assert exc.value.code == "no_category_columns"


# --------------------------------------------------------------------------- #
# read_wordlist — semantics
# --------------------------------------------------------------------------- #
def test_read_wordlist_terms_are_lowercased(tmp_path: Path) -> None:
    df = pd.DataFrame({"DicTerm": ["Happy", "SAD"], "Affect": ["X", "X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert result["Affect"].exact_single == frozenset({"happy", "sad"})


def test_read_wordlist_active_cell_is_case_insensitive(tmp_path: Path) -> None:
    df = pd.DataFrame({"DicTerm": ["happy", "sad"], "Affect": ["x", "  X  "]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert result["Affect"].exact_single == frozenset({"happy", "sad"})


def test_read_wordlist_empty_term_skipped(tmp_path: Path) -> None:
    df = pd.DataFrame({"DicTerm": ["", "happy"], "Affect": ["X", "X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert result["Affect"].exact_single == frozenset({"happy"})


def test_read_wordlist_whitespace_term_skipped(tmp_path: Path) -> None:
    # A whitespace-only term strips to empty (but is not NaN) -> skipped.
    df = pd.DataFrame({"DicTerm": ["   ", "happy"], "Affect": ["X", "X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert result["Affect"].exact_single == frozenset({"happy"})


def test_read_wordlist_bare_star_skipped(tmp_path: Path) -> None:
    # A term that is just "*" -> empty prefix -> skipped.
    df = pd.DataFrame({"DicTerm": ["*", "happy"], "Affect": ["X", "X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert result["Affect"].wildcard_single == ()
    assert result["Affect"].exact_single == frozenset({"happy"})


def test_read_wordlist_multiword_wildcard_strips_star(tmp_path: Path) -> None:
    df = pd.DataFrame({"DicTerm": ["new *"], "Place": ["X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert result["Place"].wildcard_multi == ("new",)


def test_read_wordlist_sanitizes_category_names(tmp_path: Path) -> None:
    df = pd.DataFrame({"DicTerm": ["happy"], "Affect-Modal": ["X"], "Place/Location": ["X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert set(result) == {"Affect_Modal", "Place_Location"}


def test_read_wordlist_uniquifies_collision_categories(tmp_path: Path) -> None:
    # Two categories differing only in stripped chars -> sanitized then uniquified.
    df = pd.DataFrame({"DicTerm": ["happy"], "a.b": ["X"], "a-b": ["X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert len(result) == 2
    assert set(result) == {"a_b", "a_b_2"}


def test_read_wordlist_term_in_multiple_categories(tmp_path: Path) -> None:
    df = pd.DataFrame({"DicTerm": ["happy"], "Affect": ["X"], "Positive": ["X"]})
    path = _write_csv(tmp_path / "wl.csv", df)
    result = read_wordlist(path)
    assert "happy" in result["Affect"].exact_single
    assert "happy" in result["Positive"].exact_single


# --------------------------------------------------------------------------- #
# merge_wordlists — namespacing + global uniqueness (§10)
# --------------------------------------------------------------------------- #
def test_merge_wordlists_namespaces_categories(tmp_path: Path) -> None:
    p1 = _write_csv(tmp_path / "dic1.csv", WORDLIST_DF)
    p2 = _write_csv(
        tmp_path / "dic2.csv",
        pd.DataFrame({"DicTerm": ["angry"], "Affect": ["X"]}),
    )
    result = merge_wordlists([p1, p2])
    assert len(result) == 2
    assert all(isinstance(w, Wordlist) for w in result)

    all_cats = {c for w in result for c in w.category_names}
    # Both have an "Affect" category -> namespaced apart, no collision.
    assert "dic1_Affect" in all_cats
    assert "dic2_Affect" in all_cats
    assert len(all_cats) == len(set(all_cats))  # globally unique


def test_merge_wordlists_uses_prefixes(tmp_path: Path) -> None:
    p1 = _write_csv(tmp_path / "dic1.csv", WORDLIST_DF)
    result = merge_wordlists([p1], prefixes=["custom"])
    assert result[0].namespace == "custom"
    assert "custom_Affect" in result[0].category_names


def test_merge_wordlist_prefix_none_falls_back_to_stem(tmp_path: Path) -> None:
    p1 = _write_csv(tmp_path / "mydic.csv", WORDLIST_DF)
    result = merge_wordlists([p1], prefixes=[None])
    assert result[0].namespace == "mydic"
    assert "mydic_Affect" in result[0].category_names


def test_merge_wordlists_mapping_prefixes(tmp_path: Path) -> None:
    p1 = _write_csv(tmp_path / "dic1.csv", WORDLIST_DF)
    result = merge_wordlists([p1], prefixes={"dic1.csv": "emo"})
    assert result[0].namespace == "emo"
    assert "emo_Affect" in result[0].category_names


def test_merge_wordlists_global_uniquify_across_files(tmp_path: Path) -> None:
    # Two files with the SAME stem -> namespaces collide -> uniquified.
    sub1 = tmp_path / "a"
    sub2 = tmp_path / "b"
    sub1.mkdir()
    sub2.mkdir()
    p1 = _write_csv(sub1 / "dic.csv", WORDLIST_DF)
    p2 = _write_csv(sub2 / "dic.csv", WORDLIST_DF)
    result = merge_wordlists([p1, p2])
    all_cats = [c for w in result for c in w.category_names]
    assert len(all_cats) == len(set(all_cats))


def test_merge_wordlists_empty() -> None:
    assert merge_wordlists([]) == []


def test_merge_wordlists_short_prefix_list_padded(tmp_path: Path) -> None:
    p1 = _write_csv(tmp_path / "dic1.csv", WORDLIST_DF)
    p2 = _write_csv(
        tmp_path / "dic2.csv",
        pd.DataFrame({"DicTerm": ["angry"], "Affect": ["X"]}),
    )
    # Only one prefix supplied for two files -> second falls back to stem.
    result = merge_wordlists([p1, p2], prefixes=["first"])
    assert result[0].namespace == "first"
    assert result[1].namespace == "dic2"


def test_merge_wordlists_short_names_list_padded(tmp_path: Path) -> None:
    # A short names list is padded with None; for path sources None just means
    # "derive from the path", so padding is harmless.
    p1 = _write_csv(tmp_path / "dic1.csv", WORDLIST_DF)
    p2 = _write_csv(
        tmp_path / "dic2.csv",
        pd.DataFrame({"DicTerm": ["angry"], "Affect": ["X"]}),
    )
    result = merge_wordlists([p1, p2], prefixes=["a", "b"], names=["x.csv"])
    assert result[0].namespace == "a"
    assert result[1].namespace == "b"


# --------------------------------------------------------------------------- #
# Integration with counting — the loaded wordlist is analysis-ready
# --------------------------------------------------------------------------- #
def test_loaded_wordlist_drives_counting(tmp_path: Path) -> None:
    path = _write_csv(tmp_path / "wl.csv", WORDLIST_DF)
    wordlists = merge_wordlists([path])
    config = build_analysis_config(wordlists, max_n=3)
    counts = count_document("happy happy new york", config)
    # exact_single "happy" counts 2 occurrences (token frequency, §2.1 fix).
    # NOTE: the wordlist also has the wildcard prefix "happ" (from "happ*"),
    # which additionally matches "happy" x2 — so Affect's total is 4. This
    # overlap is the preserved legacy behavior: a term can be counted by both
    # its exact and wildcard entries. Locks the double-count in.
    assert counts.category_counts["wl_Affect"] == 4
    # exact_multi "new york" x1 AND wildcard_multi "new" (from "new *") matches
    # the "new york" n-gram x1 -> Place total is 2 (same overlap behavior).
    assert counts.category_counts["wl_Place"] == 2
