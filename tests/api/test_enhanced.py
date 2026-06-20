"""Enhanced dataset inspection: columns + paginated records."""

from __future__ import annotations

import httpx
import pytest
from tests.api.conftest import (
    DATASET_CSV,
    WORDLIST_CSV,
    run_analysis,
    upload_dataset,
    upload_wordlist,
)


@pytest.mark.asyncio
async def test_enhanced_columns(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(f"/analysis/{job['job_id']}/enhanced/columns")
    assert resp.status_code == 200
    cols = resp.json()["columns"]
    # Originals + n_tokens, n_types, + wl1_Affect_{word_count,word_perc,detected_words}.
    for original in ("id", "text", "group", "score"):
        assert original in cols
    assert "n_tokens" in cols
    assert "n_types" in cols
    assert "wl1_Affect_word_count" in cols
    assert "wl1_Affect_word_perc" in cols
    assert "wl1_Affect_detected_words" in cols


@pytest.mark.asyncio
async def test_enhanced_records_paginated(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(f"/analysis/{job['job_id']}/enhanced", params={"offset": 0, "limit": 2})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 6
    assert body["offset"] == 0
    assert body["limit"] == 2
    assert len(body["records"]) == 2


@pytest.mark.asyncio
async def test_enhanced_records_offset(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(
        f"/analysis/{job['job_id']}/enhanced", params={"offset": 4, "limit": 10}
    )
    body = resp.json()
    assert len(body["records"]) == 2  # rows 5 and 6


@pytest.mark.asyncio
async def test_enhanced_records_column_selection(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(
        f"/analysis/{job['job_id']}/enhanced",
        params={"columns": ["id", "n_tokens"]},
    )
    body = resp.json()
    assert body["columns"] == ["id", "n_tokens"]
    assert set(body["records"][0].keys()) == {"id", "n_tokens"}


@pytest.mark.asyncio
async def test_enhanced_not_ready_is_404_before_job(client: httpx.AsyncClient) -> None:
    resp = await client.get("/analysis/nope/enhanced/columns")
    assert resp.status_code == 404


async def _seed(client: httpx.AsyncClient) -> dict:
    ds = await upload_dataset(client, DATASET_CSV)
    wl = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    return await run_analysis(client, ds["id"], wl["ids"])
