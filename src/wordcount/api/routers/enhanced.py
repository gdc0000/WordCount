"""Enhanced dataset inspection endpoints.

GET /analysis/{job_id}/enhanced/columns — column list.
GET /analysis/{job_id}/enhanced          — paginated JSON records.

Never dumps the whole frame in one response (the enhanced frame can be large).
NaN cells become JSON ``null``; numpy scalars are coerced for JSON.
"""

from __future__ import annotations

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, Query
from fastapi.encoders import jsonable_encoder

from wordcount.api.deps import get_job_store
from wordcount.api.schemas import EnhancedColumns, EnhancedRecords
from wordcount.api.storage import JobStore

router = APIRouter(prefix="/analysis", tags=["enhanced"])

StoreDep = Annotated[JobStore, Depends(get_job_store)]


def _enhanced_df(store: JobStore, job_id: str) -> pd.DataFrame:
    job = store.get(job_id)  # KeyError → 404
    if job.enhanced_df is None:
        raise KeyError(f"enhanced result for job {job_id!r} not ready")
    return job.enhanced_df


@router.get("/{job_id}/enhanced/columns", response_model=EnhancedColumns)
def enhanced_columns(
    job_id: str,
    store: StoreDep,
) -> EnhancedColumns:
    """List the enhanced dataset's columns."""
    df = _enhanced_df(store, job_id)
    return EnhancedColumns(columns=list(df.columns))


@router.get("/{job_id}/enhanced", response_model=EnhancedRecords)
def enhanced_records(
    job_id: str,
    store: StoreDep,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int | None, Query(ge=1)] = None,
    columns: Annotated[list[str] | None, Query()] = None,
) -> EnhancedRecords:
    """Paginated JSON records of the enhanced dataset."""
    df = _enhanced_df(store, job_id)
    selected = list(df.columns) if columns is None else columns
    view = df[selected]
    end = None if limit is None else offset + limit
    page = view.iloc[offset:end]
    # NaN → None so the wire shape is valid JSON; jsonable_encoder handles numpy.
    page = page.where(pd.notna(page), None)
    records = jsonable_encoder(page.to_dict(orient="records"))
    total = len(df)
    return EnhancedRecords(
        offset=offset,
        limit=limit,
        total=total,
        columns=selected,
        records=records,
    )


__all__ = ["router"]
