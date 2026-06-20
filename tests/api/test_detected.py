"""Detected terms — the §2.2 regression surface (structured counts)."""

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
async def test_detected_terms_carry_frequency_not_doc_count(client: httpx.AsyncClient) -> None:
    """§2.2 headline: frequency = total token occurrences, not doc count.

    'happy' appears: row1 once, row3 twice, row5 once = 4 occurrences across 3
    docs. Plus 'happ*' (wildcard→prefix happ) also matches 'happy' — but that's
    a separate detected entry; the exact 'happy' entry must show frequency 4.
    """
    job = await _seed(client)
    resp = await client.get(f"/analysis/{job['job_id']}/detected/wl1_Affect", params={"top_n": 20})
    assert resp.status_code == 200
    body = resp.json()
    assert body["category"] == "wl1_Affect"
    terms = {t["term"]: t for t in body["terms"]}
    # 'happy' exact: 1 + 2 + 1 = 4 occurrences in 3 docs.
    assert terms["happy"]["frequency"] == 4
    assert terms["happy"]["doc_frequency"] == 3
    # 'glad': row1 once, row5 twice = 3 occurrences in 2 docs.
    assert terms["glad"]["frequency"] == 3
    assert terms["glad"]["doc_frequency"] == 2


@pytest.mark.asyncio
async def test_detected_terms_top_n(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(f"/analysis/{job['job_id']}/detected/wl1_Affect", params={"top_n": 1})
    body = resp.json()
    assert body["top_n"] == 1
    assert len(body["terms"]) == 1


@pytest.mark.asyncio
async def test_detected_unknown_category_is_404(client: httpx.AsyncClient) -> None:
    job = await _seed(client)
    resp = await client.get(f"/analysis/{job['job_id']}/detected/Nope", params={"top_n": 5})
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


async def _seed(client: httpx.AsyncClient) -> dict:
    ds = await upload_dataset(client, DATASET_CSV)
    wl = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    return await run_analysis(client, ds["id"], wl["ids"])
