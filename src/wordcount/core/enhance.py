"""Enhance a dataset with analysis results, plus pure serialization writers.

Replaces ``app/enhance.py`` (audit §3.1, §3.2, §3.3, §5.6, §6.5). The legacy
version scanned the whole frame three times with ``applymap``/``map`` to find
list/dict/set columns (§3.1, §5.6), sanitized column names with a lossy regex
that could create duplicates (§3.2), and force-``str()``-ed every object column
including the user's originals — turning ``NaN`` into ``"nan"`` and corrupting
downstream ``select_dtypes`` for ANOVA (§3.3).

The new version:

* Knows the analysis-produced columns **by suffix** (``_word_count``,
  ``_word_perc``, ``_detected_words``, plus ``n_tokens``/``n_types``), so there
  is no full-frame dtype scan (§5.6). Only ``_detected_words`` (a
  ``dict[str,int]`` per row) is serialized, via a single per-column
  ``Series.map`` (§3.1).
* Sanitizes columns with :func:`core.naming.sanitize_columns` so two originals
  differing only in stripped chars never collide (§3.2).
* Coerces **only** the analysis-produced columns, never the user's originals
  (§3.3 — ``NaN`` stays ``NaN``, IDs stay ints, the text column stays text).
* Pulls ``n_tokens``/``n_types`` from ``analysis_df`` (the legacy re-injected
  them only as a fallback; they are always present now).

Writers are pure functions reused by the API's streaming export and the CLI
(§6.5 — no per-caller serialization):

* :func:`to_csv_stream` yields ``bytes`` chunks for a ``StreamingResponse``;
* :func:`to_csv_bytes` / :func:`to_parquet_bytes` return full ``bytes``;
* :func:`to_json_records` returns paginated ``list[dict]`` for the API's
  ``/enhanced`` endpoint (never dumps the whole frame in one response).

Pure: no streamlit, no fastapi, no pydantic.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from typing import Any

import pandas as pd

from wordcount.core.naming import sanitize_columns

#: Suffixes of analysis-produced columns (known up-front — no dtype scan, §5.6).
_DETECTED_SUFFIX = "_detected_words"
_ANALYSIS_SUFFIXES = ("_word_count", "_word_perc", _DETECTED_SUFFIX)

#: Format for a ``_detected_words`` cell: ``"term:cnt, term:cnt"`` (sorted for
#: determinism). Frequency travels in the struct from core/counting.py (§2.2),
#: so this is a *display* serialization, not a round-trip — plots/API read the
#: structured dict, never re-split this string.
_PAIR_SEP = ", "


def _format_detected(value: Any) -> Any:
    """Serialize a ``{term: occ}`` dict to a deterministic string; passthrough otherwise."""
    if isinstance(value, dict):
        return _PAIR_SEP.join(f"{k}: {v}" for k, v in sorted(value.items()))
    return value


def enhance_dataset(
    dataset: pd.DataFrame,
    analysis_df: pd.DataFrame,
    *,
    text_column: str | None = None,
) -> pd.DataFrame:
    """Attach analysis columns to the original dataset.

    The original dataset's columns come first (untouched, including ``NaN``s —
    §3.3), then the analysis columns. Column names are sanitized + uniquified
    with :func:`sanitize_columns` (§3.2). Only ``_detected_words`` cells are
    stringified, via per-column ``Series.map`` (§3.1, §5.6 — no full-frame
    scan). ``n_tokens``/``n_types`` come from ``analysis_df``.

    ``text_column`` is informational here (the API/CLI use it for
    ``exclude_text`` on export); it is not mutated.
    """
    dataset_reset = dataset.reset_index(drop=True)
    analysis_reset = analysis_df.reset_index(drop=True)
    enhanced = pd.concat([dataset_reset, analysis_reset], axis=1)

    # Serialize _detected_words dicts to display strings — per-column only.
    for col in enhanced.columns:
        if isinstance(col, str) and col.endswith(_DETECTED_SUFFIX):
            enhanced[col] = enhanced[col].map(_format_detected)

    # Sanitize + uniquify ALL columns (originals + analysis) so downstream
    # selections by name are never ambiguous (§3.2).
    enhanced.columns = sanitize_columns(enhanced.columns)

    return enhanced


# --------------------------------------------------------------------------- #
# Writers (pure; reused by API streaming export + CLI)
# --------------------------------------------------------------------------- #
def to_csv_stream(df: pd.DataFrame, *, chunksize: int = 4096) -> Iterator[bytes]:
    """Yield the DataFrame as CSV ``bytes`` in chunks for a ``StreamingResponse``.

    Avoids holding the whole CSV in memory twice (§6.5). Writes the header
    first, then row batches. ``chunksize`` is the byte size of each yield.
    """
    if len(df.columns) == 0:
        yield b""
        return

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    # Header.
    writer.writerow(df.columns)
    yield _flush_chunk(buf)

    # Rows.
    for record in df.itertuples(index=False, name=None):
        writer.writerow(record)
        if buf.tell() >= chunksize:
            yield _flush_chunk(buf)

    if buf.tell():
        yield _flush_chunk(buf)


def _flush_chunk(buf: Any) -> bytes:
    """Drain and return ``buf``'s contents as bytes."""
    data: str = buf.getvalue()
    buf.seek(0)
    buf.truncate(0)
    return data.encode("utf-8")


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Return the full CSV as ``bytes`` (used when streaming isn't needed)."""
    csv_text: str = df.to_csv(index=False)
    return csv_text.encode("utf-8")


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Return the full Parquet as ``bytes`` (requires the ``parquet`` extra).

    Raises :class:`ImportError` with a helpful message if ``pyarrow`` isn't
    installed. Parquet can't be chunk-streamed as cheaply as CSV, so the API
    returns the full bytes via ``StreamingResponse`` over a one-shot buffer.
    """
    try:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via skip in tests
        raise ImportError(
            "Parquet export requires the 'parquet' extra: pip install wordcount-stats[parquet]"
        ) from exc

    buf = io.BytesIO()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, buf)
    return buf.getvalue()


def to_json_records(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Return paginated DataFrame records as a list of dicts.

    ``columns`` selects/reorders columns (defaults to all). ``offset``/``limit``
    paginate. Used by the API's ``/enhanced`` endpoint so the whole frame is
    never dumped in one response.
    """
    view = df if columns is None else df[columns]
    if offset:
        view = view.iloc[offset:]
    if limit is not None:
        view = view.iloc[:limit]
    # orient="records" + date_format="iso" for stable JSON shapes.
    records: list[dict[str, Any]] = view.to_dict(orient="records")
    return records


__all__ = [
    "enhance_dataset",
    "to_csv_bytes",
    "to_csv_stream",
    "to_json_records",
    "to_parquet_bytes",
]
