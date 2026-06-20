"""Dataset upload/inspect/delete."""

from __future__ import annotations

import httpx
import pytest
from tests.api.conftest import DATASET_CSV


@pytest.mark.asyncio
async def test_upload_dataset_returns_meta_and_preview(client: httpx.AsyncClient) -> None:
    body = await _upload(client)
    assert body["name"] == "data.csv"
    assert body["n_rows"] == 6
    assert body["columns"] == ["id", "text", "group", "score"]
    assert body["dtype_guesses"]["score"] == "int64"
    assert len(body["preview"]) == 5  # capped at 5
    assert body["preview"][0]["text"] == "I am happy and glad"


@pytest.mark.asyncio
async def test_get_dataset(client: httpx.AsyncClient) -> None:
    uploaded = await _upload(client)
    resp = await client.get(f"/datasets/{uploaded['id']}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == uploaded["id"]
    assert body["n_rows"] == 6


@pytest.mark.asyncio
async def test_delete_dataset(client: httpx.AsyncClient) -> None:
    uploaded = await _upload(client)
    resp = await client.delete(f"/datasets/{uploaded['id']}")
    assert resp.status_code == 204
    # Subsequent get -> 404 ProblemDetails.
    resp = await client.get(f"/datasets/{uploaded['id']}")
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


@pytest.mark.asyncio
async def test_get_unknown_dataset_is_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/datasets/does-not-exist")
    assert resp.status_code == 404
    body = resp.json()
    assert body["type"].endswith("/not_found")
    assert body["status"] == 404


@pytest.mark.asyncio
async def test_upload_unsupported_format_is_415(client: httpx.AsyncClient) -> None:
    resp = await client.post(
        "/datasets",
        files={"file": ("data.foo", b"garbage", "application/octet-stream")},
    )
    assert resp.status_code == 415
    body = resp.json()
    assert body["code"] == "unsupported_format"
    assert body["status"] == 415


async def _upload(client: httpx.AsyncClient) -> dict:
    resp = await client.post(
        "/datasets",
        files={"file": ("data.csv", DATASET_CSV.encode(), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()
