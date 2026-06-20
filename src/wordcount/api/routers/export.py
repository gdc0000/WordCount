"""Streaming export endpoint.

GET /export/{job_id}?format=csv|parquet&exclude_text=true — chunked download.

CSV is streamed via :func:`core.enhance.to_csv_stream` (yields bytes chunks,
never holds the whole CSV in memory twice — fixes §6.5). Parquet returns the
full bytes (pyarrow can't cheaply chunk-stream) over a one-shot buffer.
``exclude_text`` drops the job's text column from the export.
"""

from __future__ import annotations

from typing import Annotated, Literal

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from wordcount.api.deps import get_job_store
from wordcount.api.storage import JobStore
from wordcount.core.enhance import to_csv_stream, to_parquet_bytes

router = APIRouter(prefix="/export", tags=["export"])

StoreDep = Annotated[JobStore, Depends(get_job_store)]


def _enhanced_df(store: JobStore, job_id: str, exclude_text: bool) -> pd.DataFrame:
    job = store.get(job_id)  # KeyError → 404
    if job.enhanced_df is None:
        raise KeyError(f"enhanced result for job {job_id!r} not ready")
    df = job.enhanced_df
    if exclude_text and job.text_column and job.text_column in df.columns:
        df = df.drop(columns=[job.text_column])
    return df


@router.get("/{job_id}")
def export(
    job_id: str,
    store: StoreDep,
    format: Annotated[Literal["csv", "parquet"], Query()] = "csv",
    exclude_text: Annotated[bool, Query()] = False,
) -> StreamingResponse:
    """Download an enhanced dataset as streamed CSV or Parquet bytes."""
    df = _enhanced_df(store, job_id, exclude_text)
    if format == "csv":
        return StreamingResponse(
            to_csv_stream(df),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=enhanced_{job_id}.csv"},
        )
    # Parquet path (requires the `parquet` extra).
    try:
        payload = to_parquet_bytes(df)
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    return StreamingResponse(
        iter([payload]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=enhanced_{job_id}.parquet"},
    )


__all__ = ["router"]
