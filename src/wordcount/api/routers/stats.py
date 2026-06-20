"""Stats endpoints — thin DTO wrappers over pure ``core.stats``.

POST /stats/pearson — Pearson correlation between two enhanced columns.
POST /stats/anova   — one-way ANOVA with Tukey HSD + t-CI group descriptives.

The math is entirely in :mod:`core.stats` (p-value by row name §2.3, t-CI §2.5).
These routes do only: parse the request → call core → shape the result (Pydantic)
→ return. The ANOVA table (a DataFrame) is serialized as records so the wire
shape is JSON-stable.
"""

from __future__ import annotations

import json
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends

from wordcount.api.deps import get_job_store
from wordcount.api.schemas import (
    AnovaRequest,
    AnovaResponse,
    GroupStatsDTO,
    PearsonRequest,
    PearsonResponse,
)
from wordcount.api.storage import JobStore
from wordcount.core.stats import anova, pearson

router = APIRouter(prefix="/stats", tags=["stats"])

StoreDep = Annotated[JobStore, Depends(get_job_store)]


def _enhanced_df(store: JobStore, job_id: str) -> pd.DataFrame:
    job = store.get(job_id)  # KeyError → 404
    if job.enhanced_df is None:
        raise KeyError(f"enhanced result for job {job_id!r} not ready")
    return job.enhanced_df


@router.post("/pearson", response_model=PearsonResponse)
def pearson_endpoint(
    req: PearsonRequest,
    store: StoreDep,
) -> PearsonResponse:
    """Pearson correlation between two numeric columns of an enhanced dataset."""
    df = _enhanced_df(store, req.job_id)
    result = pearson(df, req.col1, req.col2)
    return PearsonResponse(
        col1=result.col1,
        col2=result.col2,
        coefficient=result.coefficient,
        p_value=result.p_value,
        n=result.n,
    )


def _records_to_native(records: list[dict[str, object]]) -> list[dict[str, object]]:
    """Coerce numpy-typed record values to JSON-native Python via pandas.

    The ANOVA table and Tukey summary come back from statsmodels with numpy
    scalars (numpy.bool_, numpy.float64) that FastAPI's encoder rejects; this
    round-trips through ``DataFrame.to_json`` to nativize them.
    """
    if not records:
        return []
    nativized: list[dict[str, object]] = json.loads(pd.DataFrame(records).to_json(orient="records"))
    return nativized


@router.post("/anova", response_model=AnovaResponse)
def anova_endpoint(
    req: AnovaRequest,
    store: StoreDep,
) -> AnovaResponse:
    """One-way ANOVA of a numeric column across a categorical column's groups."""
    df = _enhanced_df(store, req.job_id)
    result = anova(df, req.cat_var, req.num_var)
    table_records: list[dict[str, object]] = json.loads(
        result.table.reset_index().to_json(orient="records")
    )
    return AnovaResponse(
        cat_var=result.cat_var,
        num_var=result.num_var,
        p_value=result.p_value,
        significant=result.significant,
        group_stats=[
            GroupStatsDTO(
                category=g.category,
                n=g.n,
                mean=g.mean,
                sem=g.sem,
                ci_lower=g.ci_lower,
                ci_upper=g.ci_upper,
            )
            for g in result.group_stats
        ],
        tukey_rows=_records_to_native([dict(row) for row in result.tukey_rows]),
        table=table_records,
    )


__all__ = ["router"]
