"""Pure plot-data builders (no rendering, no streamlit).

Replaces ``app/plots.py:generate_barplot`` (audit §2.2). The legacy version
re-split a `", "`-joined detected-words string with ``Counter`` — lossy and
fragile (any term containing `", "` broke it). Frequency now travels in the
structured ``{term: occ}`` dicts produced by :mod:`core.counting`, so
:func:`build_barplot_data` aggregates those directly.

Two entry points, both returning long-form DataFrames ready for any plot
library (the API returns them as JSON; ``viz/charts.py`` wraps them in
Plotly figures):

* :func:`build_barplot_data` — top-N detected terms for one category, with
  ``term``, ``frequency`` (total occurrences across all docs), and
  ``doc_frequency`` (number of docs containing the term).
* :func:`aggregate_detected` — the full term->(frequency, doc_frequency)
  mapping for one category, for callers that want the raw aggregates.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import pandas as pd

#: A cell in a ``_detected_words`` column. ``core.enhance`` serializes dicts to
#: ``"term: cnt, ..."`` strings for *display*; this module reads the STRUCTURED
#: dict form that lives in the analysis DataFrame before enhancement, or the
#: API's structured representation. Both forms are accepted.
DetectedCell = dict[str, int] | str


def _cell_to_counter(cell: DetectedCell) -> Counter[str]:
    """Read one detected-words cell into a ``Counter[str, int]``.

    Accepts the structured ``{term: occ}`` dict (the canonical form from
    :mod:`core.counting`) or, as a fallback for already-enhanced frames, the
    ``"term: cnt, ..."`` display string. The dict path is the source of truth
    (§2.2: no lossy string round-trip); the string path is a tolerant fallback
    for callers holding an enhanced frame.
    """
    if isinstance(cell, dict):
        return Counter({str(k): int(v) for k, v in cell.items()})
    if isinstance(cell, str) and cell:
        counter: Counter[str] = Counter()
        for pair in cell.split(", "):
            if not pair.strip():
                continue
            if ": " in pair:
                term, _, cnt = pair.rpartition(": ")
                try:
                    counter[term.strip()] += int(cnt)
                except ValueError:
                    continue
            else:
                # Tolerate the legacy ", "-joined list form (no counts).
                counter[pair.strip()] += 1
        return counter
    return Counter()


def aggregate_detected(
    detected_series: pd.Series, cells: Sequence[DetectedCell] | None = None
) -> tuple[list[str], list[int], list[int]]:
    """Aggregate detected terms across all documents for one category.

    Pass either a ``pd.Series`` of detected-words cells (from the analysis
    DataFrame) or a ``cells`` sequence. Returns three parallel lists:
    ``terms``, ``frequency`` (total occurrences), ``doc_frequency`` (number of
    documents in which the term appears). Terms are sorted by descending
    frequency then alphabetically for determinism.
    """
    source = cells if cells is not None else detected_series.tolist()
    freq: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    for cell in source:
        if pd.isna(cell):
            continue
        per_doc = _cell_to_counter(cell)
        for term, occ in per_doc.items():
            freq[term] += occ
            doc_freq[term] += 1
    # Sort by descending frequency, then term for determinism.
    ordered = sorted(freq.keys(), key=lambda t: (-freq[t], t))
    return (
        ordered,
        [freq[t] for t in ordered],
        [doc_freq[t] for t in ordered],
    )


def build_barplot_data(
    detected_series: pd.Series,
    *,
    top_n: int = 10,
    cells: Sequence[DetectedCell] | None = None,
) -> pd.DataFrame:
    """Long-form DataFrame of the top-N detected terms for one category.

    Columns: ``term``, ``frequency`` (total occurrences), ``doc_frequency``
    (number of documents containing the term). Sorted by descending frequency
    then term. Reads the structured ``{term: occ}`` dict directly — no string
    re-split (fixes §2.2).
    """
    terms, frequencies, doc_frequencies = aggregate_detected(detected_series, cells)
    terms = terms[:top_n]
    frequencies = frequencies[:top_n]
    doc_frequencies = doc_frequencies[:top_n]
    return pd.DataFrame(
        {
            "term": terms,
            "frequency": frequencies,
            "doc_frequency": doc_frequencies,
        }
    )


__all__ = ["aggregate_detected", "build_barplot_data"]
