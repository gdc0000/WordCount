"""Wordlist upload + summary."""

from __future__ import annotations

import httpx
import pytest

from tests.api.conftest import WORDLIST_CSV, upload_wordlist


@pytest.mark.asyncio
async def test_upload_wordlist_namespaces_and_summary(client: httpx.AsyncClient) -> None:
    body = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    assert len(body["ids"]) == 1
    assert body["namespaces"] == ["wl1"]
    assert body["categories_per_list"] == [1]
    summary = body["summary"][0]
    assert summary["namespace"] == "wl1"
    cat = summary["categories"][0]
    # Namespaced category: wl1_Affect.
    assert cat["category"] == "wl1_Affect"
    # Terms: happy, glad, sad, angry (exact single) + happ (wildcard single).
    assert cat["n_terms"] == 5
    samples = set(cat["sample_terms"])
    assert {"happy", "glad", "sad", "angry"}.issubset(samples)
    assert "happ*" in samples  # wildcard rendered with its star


@pytest.mark.asyncio
async def test_wordlist_summary_endpoint(client: httpx.AsyncClient) -> None:
    uploaded = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    wid = uploaded["ids"][0]
    resp = await client.get(f"/wordlists/{wid}/summary")
    assert resp.status_code == 200
    summary = resp.json()
    assert summary["namespace"] == "wl1"
    assert summary["categories"][0]["category"] == "wl1_Affect"


@pytest.mark.asyncio
async def test_wordlist_missing_dicterm_is_422(client: httpx.AsyncClient) -> None:
    bad = "Term,Affect\nfoo,X\n"
    resp = await client.post(
        "/wordlists",
        files=[("files", ("wl.csv", bad.encode(), "text/csv"))],
        data={"prefixes": "wl1"},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["code"] == "missing_dicterm_column"


@pytest.mark.asyncio
async def test_wordlist_unknown_id_is_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/wordlists/nope/summary")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
