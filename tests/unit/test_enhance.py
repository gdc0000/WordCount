"""Tests for ``wordcount.core.enhance`` — the §3.1/§3.2/§3.3/§5.6/§6.5 fixes
and the pure writers (CSV stream/bytes, Parquet bytes, paginated JSON records).

Pure core tests: pandas builds fixtures; no streamlit, no fastapi.
"""

from __future__ import annotations

import io as _io
from typing import Any

import numpy as np
import pandas as pd
import pytest

from wordcount.core.enhance import (
    enhance_dataset,
    to_csv_bytes,
    to_csv_stream,
    to_json_records,
    to_parquet_bytes,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# --------------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------------- #
def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "text": ["happy day", "sad sad night", "new york baby"],
            "score": [0.1, 0.2, 0.3],
            "maybe_missing": [np.nan, "x", np.nan],
        }
    )


def _analysis() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "n_tokens": [2, 3, 3],
            "n_types": [2, 2, 3],
            "Affect_word_count": [1, 2, 0],
            "Affect_word_perc": [0.5, 2 / 3, 0.0],
            "Affect_detected_words": [{"happy": 1}, {"sad": 2}, {}],
            "Place_word_count": [0, 0, 1],
            "Place_word_perc": [0.0, 0.0, 1 / 3],
            "Place_detected_words": [{}, {}, {"new york": 1}],
        }
    )


# --------------------------------------------------------------------------- #
# enhance_dataset — column layout
# --------------------------------------------------------------------------- #
def test_enhance_preserves_originals_then_appends_analysis() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    cols = list(enhanced.columns)
    # Originals first (untouched), then analysis columns.
    assert cols[:4] == ["id", "text", "score", "maybe_missing"]
    assert "n_tokens" in cols and "n_types" in cols
    assert "Affect_word_count" in cols


def test_enhance_row_count_matches_dataset() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    assert len(enhanced) == 3


def test_enhance_n_tokens_n_types_from_analysis() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    assert list(enhanced["n_tokens"]) == [2, 3, 3]
    assert list(enhanced["n_types"]) == [2, 2, 3]


# --------------------------------------------------------------------------- #
# §3.1 + §5.6 — only _detected_words serialized, per-column, no dtype scan
# --------------------------------------------------------------------------- #
def test_enhance_serializes_detected_words_to_string() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    # dict -> "term: cnt, ..." deterministic string.
    assert enhanced.loc[0, "Affect_detected_words"] == "happy: 1"
    assert enhanced.loc[1, "Affect_detected_words"] == "sad: 2"
    assert enhanced.loc[2, "Affect_detected_words"] == ""


def test_enhance_detected_words_sorted_for_determinism() -> None:
    analysis = pd.DataFrame(
        {
            "n_tokens": [1],
            "n_types": [1],
            "Cat_word_count": [3],
            "Cat_word_perc": [1.0],
            "Cat_detected_words": [{"zebra": 1, "apple": 2}],
        }
    )
    enhanced = enhance_dataset(pd.DataFrame({"x": [1]}), analysis)
    assert enhanced.loc[0, "Cat_detected_words"] == "apple: 2, zebra: 1"


def test_enhance_detected_words_non_dict_passthrough() -> None:
    # A non-dict value in a _detected_words column is left as-is by the mapper.
    analysis = pd.DataFrame(
        {
            "n_tokens": [1, 1],
            "n_types": [1, 1],
            "Cat_word_count": [1, 0],
            "Cat_word_perc": [1.0, 0.0],
            "Cat_detected_words": [{"happy": 1}, "already a string"],
        }
    )
    enhanced = enhance_dataset(pd.DataFrame({"x": [1, 2]}), analysis)
    assert enhanced.loc[0, "Cat_detected_words"] == "happy: 1"
    assert enhanced.loc[1, "Cat_detected_words"] == "already a string"


def test_enhance_does_not_scan_or_coerce_non_detected_object_columns() -> None:
    # The text column stays object (strings), not force-str'd; the score stays
    # float; maybe_missing stays NaN (not "nan").
    enhanced = enhance_dataset(_dataset(), _analysis())
    assert enhanced["score"].dtype == np.float64
    assert pd.isna(enhanced.loc[0, "maybe_missing"])


# --------------------------------------------------------------------------- #
# §3.3 — NaN preservation in original columns
# --------------------------------------------------------------------------- #
def test_enhance_preserves_nan_in_originals() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    assert pd.isna(enhanced.loc[0, "maybe_missing"])
    assert pd.isna(enhanced.loc[2, "maybe_missing"])
    assert enhanced.loc[1, "maybe_missing"] == "x"


def test_enhance_preserves_int_originals() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    # id stays int (not str), so downstream numeric selection works.
    assert str(enhanced["id"].dtype).startswith("int")


# --------------------------------------------------------------------------- #
# §3.2 — column sanitization + uniqueness
# --------------------------------------------------------------------------- #
def test_enhance_sanitizes_column_names() -> None:
    dataset = pd.DataFrame({"weird name!": [1, 2, 3], "text": ["a", "b", "c"]})
    enhanced = enhance_dataset(dataset, _analysis())
    assert "weird_name" in enhanced.columns
    assert "text" in enhanced.columns


def test_enhance_uniquifies_collision_columns() -> None:
    # Two originals differing only in stripped chars -> sanitized then uniquified.
    dataset = pd.DataFrame({"a.b": [1, 2, 3], "a-b": [4, 5, 6], "text": ["a", "b", "c"]})
    enhanced = enhance_dataset(dataset, _analysis())
    cols = list(enhanced.columns)
    assert len(cols) == len(set(cols))
    assert "a_b" in cols and "a_b_2" in cols


# --------------------------------------------------------------------------- #
# text_column is informational (not mutated)
# --------------------------------------------------------------------------- #
def test_enhance_text_column_does_not_mutate() -> None:
    ds = _dataset()
    enhanced = enhance_dataset(ds, _analysis(), text_column="text")
    assert list(enhanced["text"]) == ["happy day", "sad sad night", "new york baby"]


# --------------------------------------------------------------------------- #
# Writers — to_csv_bytes
# --------------------------------------------------------------------------- #
def test_to_csv_bytes_roundtrip() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    data = to_csv_bytes(df)
    assert isinstance(data, bytes)
    roundtripped = pd.read_csv(_io.BytesIO(data))
    pd.testing.assert_frame_equal(roundtripped, df)


def test_to_csv_bytes_no_index() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    data = to_csv_bytes(df).decode("utf-8")
    assert "Unnamed: 0" not in data
    assert data.splitlines()[0] == "a"


# --------------------------------------------------------------------------- #
# Writers — to_csv_stream (§6.5)
# --------------------------------------------------------------------------- #
def test_to_csv_stream_concatenates_to_full_csv() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    streamed = b"".join(to_csv_stream(df))
    # The stream uses \n line terminators (HTTP-friendly); compare parsed frames,
    # not raw bytes, so the test is independent of platform line-ending defaults.
    rt = pd.read_csv(_io.BytesIO(streamed))
    pd.testing.assert_frame_equal(rt, df)


def test_to_csv_stream_yields_bytes_chunks() -> None:
    df = pd.DataFrame({"a": list(range(100)), "b": [f"v{i}" for i in range(100)]})
    chunks = list(to_csv_stream(df, chunksize=64))
    assert all(isinstance(c, bytes) for c in chunks)
    assert len(chunks) > 1
    # Reassembles to the full CSV.
    reassembled = b"".join(chunks)
    rt = pd.read_csv(_io.BytesIO(reassembled))
    pd.testing.assert_frame_equal(rt, df)


def test_to_csv_stream_empty_dataframe() -> None:
    df = pd.DataFrame()
    chunks = list(to_csv_stream(df))
    # Empty frame -> a single empty chunk (or nothing); never errors.
    assert b"".join(chunks) == b""


def test_to_csv_stream_header_smaller_than_chunk() -> None:
    # Small frame, large chunksize: header flush produces an empty chunk via
    # _flush_chunk's early return when the buffer has no accumulated data yet.
    df = pd.DataFrame({"a": [1, 2]})
    chunks = list(to_csv_stream(df, chunksize=10**9))
    assert all(isinstance(c, bytes) for c in chunks)
    rt = pd.read_csv(_io.BytesIO(b"".join(chunks)))
    pd.testing.assert_frame_equal(rt, df)


def test_to_csv_stream_preserves_values_with_commas() -> None:
    # CSV quoting must handle commas inside cell values.
    df = pd.DataFrame({"a": ["hello, world", "plain"]})
    streamed = b"".join(to_csv_stream(df))
    rt = pd.read_csv(_io.BytesIO(streamed))
    pd.testing.assert_frame_equal(rt, df)


# --------------------------------------------------------------------------- #
# Writers — to_parquet_bytes (§6.5; skip if pyarrow missing)
# --------------------------------------------------------------------------- #
def test_to_parquet_bytes_roundtrip() -> None:
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    data = to_parquet_bytes(df)
    assert isinstance(data, bytes)
    rt = pd.read_parquet(_io.BytesIO(data))
    pd.testing.assert_frame_equal(rt, df)


def test_to_parquet_bytes_preserves_detected_string_column() -> None:
    pytest.importorskip("pyarrow")
    enhanced = enhance_dataset(_dataset(), _analysis())
    data = to_parquet_bytes(enhanced)
    rt = pd.read_parquet(_io.BytesIO(data))
    assert list(rt["Affect_detected_words"]) == ["happy: 1", "sad: 2", ""]


# --------------------------------------------------------------------------- #
# Writers — to_json_records (paginated)
# --------------------------------------------------------------------------- #
def test_to_json_records_all_rows() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    records = to_json_records(enhanced)
    assert len(records) == 3
    assert records[0]["n_tokens"] == 2
    assert records[1]["Affect_detected_words"] == "sad: 2"


def test_to_json_records_offset_limit() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    page = to_json_records(enhanced, offset=1, limit=1)
    assert len(page) == 1
    assert page[0]["n_tokens"] == 3


def test_to_json_records_columns_subset() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    records = to_json_records(enhanced, columns=["id", "n_tokens"])
    assert records == [
        {"id": 1, "n_tokens": 2},
        {"id": 2, "n_tokens": 3},
        {"id": 3, "n_tokens": 3},
    ]


def test_to_json_records_offset_beyond_end() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    assert to_json_records(enhanced, offset=100) == []


def test_to_json_records_records_are_plain_dicts() -> None:
    enhanced = enhance_dataset(_dataset(), _analysis())
    records = to_json_records(enhanced, limit=1)
    rec: dict[str, Any] = records[0]
    assert isinstance(rec, dict)
    # Values are JSON-native (no numpy types leaking in the default path).
    assert isinstance(rec["n_tokens"], int)
