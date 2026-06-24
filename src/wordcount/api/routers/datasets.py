"""Dataset upload/inspection endpoints.

POST /datasets        — upload a csv/tsv/xls/xlsx file, store the DataFrame.
GET  /datasets/{id}   — metadata + preview.
DELETE /datasets/{id} — free the stored frame.

Uploads are streamed to a temp file with the original suffix (so pandas detects
the format) then read via :func:`core.io.read_dataset`; the temp file is removed
immediately. The DataFrame lives in the ephemeral :class:`JobStore`.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import Response

from wordcount.api.deps import get_job_store
from wordcount.api.schemas import DatasetMeta, DatasetUpload
from wordcount.api.storage import JobStore
from wordcount.core.io import read_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])

#: Number of preview rows returned by upload/inspect.
_PREVIEW_ROWS = 5

StoreDep = Annotated[JobStore, Depends(get_job_store)]


def _meta(record_id: str, name: str, df: pd.DataFrame) -> DatasetUpload:
    """Build the dataset DTO from a stored frame."""
    dtype_guesses = {col: str(df[col].dtype) for col in df.columns}
    preview_df = df.head(_PREVIEW_ROWS)
    preview = preview_df.where(pd.notna(preview_df), None).to_dict(orient="records")
    return DatasetUpload(
        id=record_id,
        name=name,
        n_rows=len(df),
        columns=list(df.columns),
        dtype_guesses=dtype_guesses,
        preview=preview,
    )


def _save_upload(upload: UploadFile) -> tuple[Path, str]:
    """Persist an UploadFile to a temp file with its original suffix; return path+name."""
    original = upload.filename or "upload.csv"
    suffix = Path(original).suffix or ".csv"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="wc_dataset_")
    os.close(fd)
    path = Path(tmp_path)
    with path.open("wb") as fh:
        while True:
            chunk = upload.file.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
    return path, original


@router.post("", response_model=DatasetUpload, status_code=201)
async def upload_dataset(
    file: UploadFile,
    store: StoreDep,
) -> DatasetUpload:
    """Upload a dataset file; returns its id, columns, dtypes, and a preview."""
    path, original = _save_upload(file)
    try:
        df = read_dataset(path, name=original)
    finally:
        path.unlink(missing_ok=True)
    dataset_id = uuid.uuid4().hex
    store.put_dataset(dataset_id, original, df)
    return _meta(dataset_id, original, df)


@router.get("/{dataset_id}", response_model=DatasetMeta)
def get_dataset(
    dataset_id: str,
    store: StoreDep,
) -> DatasetUpload:
    """Return metadata + preview for a stored dataset."""
    record = store.get_dataset(dataset_id)
    return _meta(record.id, record.name, record.df)


@router.delete("/{dataset_id}", status_code=204)
def delete_dataset(
    dataset_id: str,
    store: StoreDep,
) -> Response:
    """Free a stored dataset."""
    store.delete_dataset(dataset_id)
    return Response(status_code=204)


__all__ = ["router"]
