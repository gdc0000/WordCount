"""Detected-terms endpoint (the §2.2 regression surface).

GET /analysis/{job_id}/detected/{category}?top_n= — top-N detected terms for a
category, reading the **structured** ``{term: occ}`` counts from the analysis
DataFrame (not re-splitting a joined string — fixes §2.2). ``frequency`` is
total token occurrences; ``doc_frequency`` is the number of documents in which
the term appears.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Query

from wordcount.api.deps import get_job_store
from wordcount.api.schemas import DetectedTerms, TermFreq
from wordcount.api.storage import JobStore
from wordcount.viz.plots import build_barplot_data

router = APIRouter(prefix="/analysis", tags=["detected"])

StoreDep = Annotated[JobStore, Depends(get_job_store)]


@router.get("/{job_id}/detected/{category}", response_model=DetectedTerms)
def detected_terms(
    job_id: str,
    category: str,
    store: StoreDep,
    top_n: Annotated[int, Query(ge=1)] = 10,
) -> DetectedTerms:
    """Top-N detected terms for a category (structured counts, §2.2)."""
    job = store.get(job_id)  # KeyError → 404
    if job.analysis_df is None:
        raise KeyError(f"analysis result for job {job_id!r} not ready")

    col = f"{category}_detected_words"
    if col not in job.analysis_df.columns:
        raise KeyError(f"category {category!r} has no detected-words column")

    series: pd.Series = job.analysis_df[col]
    bar_df = build_barplot_data(series, top_n=top_n)
    terms = [
        TermFreq(
            term=str(row.term),
            frequency=int(row.frequency),
            doc_frequency=int(row.doc_frequency),
        )
        for row in bar_df.itertuples(index=False)
    ]
    return DetectedTerms(category=category, top_n=top_n, terms=terms)


__all__ = ["router"]
