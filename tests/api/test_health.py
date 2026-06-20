"""GET /healthz — liveness + connector config."""

from __future__ import annotations

import httpx
import pytest


@pytest.mark.asyncio
async def test_health_reports_config(client: httpx.AsyncClient) -> None:
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["max_n"] == 3
    assert body["parallel"] == 1
    assert "version" in body
