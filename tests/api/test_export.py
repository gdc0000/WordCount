"""Streaming export: CSV (streamed) + Parquet + exclude_text."""

from __future__ import annotations

import io

import httpx
import pandas as pd
import pytest
from tests.api.conftest import (
    DATASET_CSV,
    WORDLIST_CSV,
    run_analysis,
    upload_dataset,
    upload_wordlist,
)


@pytest.mark.asyncio
async def test_export_csv_streaming(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(f"/export/{job['job_id']}", params={"format": "csv"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert "attachment" in resp.headers["content-disposition"]
    assert resp.headers["content-disposition"].endswith(".csv")
    # The streamed body parses as CSV with the enhanced columns.
    df = pd.read_csv(io.StringIO(resp.text))
    assert "n_tokens" in df.columns
    assert "wl1_Affect_word_count" in df.columns
    assert len(df) == 6


@pytest.mark.asyncio
async def test_export_csv_exclude_text(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(
        f"/export/{job['job_id']}", params={"format": "csv", "exclude_text": "true"}
    )
    assert resp.status_code == 200
    df = pd.read_csv(io.StringIO(resp.text))
    assert "text" not in df.columns  # text column dropped
    assert "score" in df.columns  # other originals kept


@pytest.mark.asyncio
async def test_export_parquet(client: httpx.AsyncClient) -> None:
    pytest.importorskip("pyarrow")
    job = await _seed(client)
    resp = await client.get(f"/export/{job['job_id']}", params={"format": "parquet"})
    assert resp.status_code == 200
    assert resp.headers["content-disposition"].endswith(".parquet")
    df = pd.read_parquet(io.BytesIO(resp.content))
    assert "n_tokens" in df.columns
    assert len(df) == 6


@pytest.mark.asyncio
async def test_export_unknown_job_is_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/export/nope", params={"format": "csv"})
    assert resp.status_code == 404


async def _seed(client) -> dict:
    ds = await upload_dataset(client, DATASET_CSV)
    wl = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    return await run_analysis(client, ds["id"], wl["ids"])
