"""Analysis job lifecycle: POST (202), GET status, WebSocket events."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from starlette.testclient import TestClient
from tests.api.conftest import (
    DATASET_CSV,
    WORDLIST_CSV,
    SyncExecutor,
    upload_dataset,
    upload_wordlist,
)

from wordcount.api.app import create_app
from wordcount.api.deps import get_job_runner, get_job_store, get_settings
from wordcount.api.routers import analysis as analysis_route
from wordcount.api.settings import Settings
from wordcount.api.storage import InMemoryJobStore


@pytest.mark.asyncio
async def test_create_analysis_returns_202_and_completes(client: httpx.AsyncClient) -> None:
    ds = await upload_dataset(client, DATASET_CSV)
    wl = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    job = await _run(client, ds["id"], wl["ids"])
    assert "job_id" in job
    status = await client.get(f"/analysis/{job['job_id']}")
    body = status.json()
    assert body["status"] == "done"
    assert body["progress"] == 1.0
    assert body["finished"] is not None


@pytest.mark.asyncio
async def test_analysis_bad_dataset_is_404(client: httpx.AsyncClient) -> None:
    wl = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    resp = await client.post(
        "/analysis",
        json={
            "dataset_id": "missing",
            "wordlist_ids": wl["ids"],
            "text_column": "text",
            "max_n": 3,
        },
    )
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"


@pytest.mark.asyncio
async def test_analysis_bad_text_column_is_422(client: httpx.AsyncClient) -> None:
    ds = await upload_dataset(client, DATASET_CSV)
    wl = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    resp = await client.post(
        "/analysis",
        json={
            "dataset_id": ds["id"],
            "wordlist_ids": wl["ids"],
            "text_column": "nope",
            "max_n": 3,
        },
    )
    assert resp.status_code == 422
    assert resp.json()["code"] == "no_text_column"


@pytest.mark.asyncio
async def test_analysis_unknown_job_status_is_404(client: httpx.AsyncClient) -> None:
    resp = await client.get("/analysis/nope")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# WebSocket — already-finished terminal emission (SyncExecutor).
# --------------------------------------------------------------------------- #
def test_ws_emits_done_when_job_already_finished(store: InMemoryJobStore) -> None:
    """With the SyncExecutor the job finishes during POST; the WS then emits a
    synthesized ``done`` event and closes (the 'already finished' branch)."""
    settings = Settings(max_n_default=3, default_parallel_workers=1)
    app = create_app(settings=settings)
    app.dependency_overrides[get_job_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_job_runner] = SyncExecutor

    with TestClient(app) as tc:
        ds = _upload_sync(tc)
        wl = _upload_wordlist_sync(tc)
        job = _run_sync(tc, ds["id"], wl["ids"])
        with tc.websocket_connect(f"/analysis/{job['job_id']}/events") as ws:
            event = ws.receive_json()
            assert event["type"] == "done"


# --------------------------------------------------------------------------- #
# WebSocket — live progress then done, via a real background executor + a
# threading.Event handshake so ordering is deterministic.
# --------------------------------------------------------------------------- #
def test_ws_streams_live_progress_then_done(store: InMemoryJobStore) -> None:
    settings = Settings(max_n_default=3, default_parallel_workers=1)
    app = create_app(settings=settings)
    app.dependency_overrides[get_job_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings

    ready = threading.Event()
    runner = ThreadPoolExecutor(max_workers=1)
    app.dependency_overrides[get_job_runner] = lambda: runner

    def slow_job(job_id: str, request: object, s: object) -> None:
        s.set_progress(job_id, 0.5, message="halfway")  # type: ignore[attr-defined]
        ready.wait(timeout=5)
        s.set_status(job_id, "done")  # type: ignore[attr-defined]

    original = analysis_route.run_analysis_job
    analysis_route.run_analysis_job = slow_job  # type: ignore[assignment]
    try:
        with TestClient(app) as tc:
            ds = _upload_sync(tc)
            wl = _upload_wordlist_sync(tc)
            job = _run_sync(tc, ds["id"], wl["ids"])
            with tc.websocket_connect(f"/analysis/{job['job_id']}/events") as ws:
                first = ws.receive_json()
                assert first["type"] == "progress"
                assert first["progress"] == 0.5
                ready.set()  # release the job to finish
                second = ws.receive_json()
                assert second["type"] == "done"
    finally:
        analysis_route.run_analysis_job = original  # type: ignore[assignment]
        runner.shutdown(wait=True)


# --------------------------------------------------------------------------- #
# Sync helpers for the TestClient-based WS tests.
# --------------------------------------------------------------------------- #
def _upload_sync(tc: TestClient) -> dict:
    resp = tc.post(
        "/datasets",
        files={"file": ("data.csv", DATASET_CSV.encode(), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _upload_wordlist_sync(tc: TestClient) -> dict:
    resp = tc.post(
        "/wordlists",
        files=[("files", ("wl.csv", WORDLIST_CSV.encode(), "text/csv"))],
        data={"prefixes": "wl1"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _run_sync(tc: TestClient, dataset_id: str, wordlist_ids: list[str]) -> dict:
    resp = tc.post(
        "/analysis",
        json={
            "dataset_id": dataset_id,
            "wordlist_ids": wordlist_ids,
            "text_column": "text",
            "max_n": 3,
        },
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


async def _run(client: httpx.AsyncClient, dataset_id: str, wordlist_ids: list[str]) -> dict:
    resp = await client.post(
        "/analysis",
        json={
            "dataset_id": dataset_id,
            "wordlist_ids": wordlist_ids,
            "text_column": "text",
            "max_n": 3,
        },
    )
    assert resp.status_code == 202, resp.text
    return resp.json()
