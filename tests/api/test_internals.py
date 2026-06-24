"""Coverage for internal seams: storage, workers, deps, app entrypoint, errors.

These exercise the defensive branches the happy-path router tests don't reach:
job-store KeyError guards, the cross-thread ``call_soon_threadsafe`` emit path,
the worker's category filter / parallel executor / error swallowing, the
dependency-injection singletons, the ``wordcount-api`` CLI entrypoint, and the
remaining ProblemDetails handlers (empty ValueError, request-validation).
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.api.conftest import (
    DATASET_CSV,
    WORDLIST_CSV,
    SyncExecutor,
    run_analysis,
    upload_dataset,
    upload_wordlist,
)
from wordcount.api import app as app_module
from wordcount.api.deps import get_job_runner, get_job_store, get_settings
from wordcount.api.errors import register_exception_handlers
from wordcount.api.routers import analysis as analysis_route
from wordcount.api.routers import export as export_route
from wordcount.api.routers.stats import _records_to_native
from wordcount.api.schemas import AnalysisRequest
from wordcount.api.settings import Settings
from wordcount.api.storage import InMemoryJobStore
from wordcount.api.workers import _counting_executor, _select_categories, run_analysis_job
from wordcount.core.io import merge_wordlists, read_dataset


# --------------------------------------------------------------------------- #
# storage.py — KeyError guards + set_error + cross-thread emit + delete miss
# --------------------------------------------------------------------------- #
def test_store_keyerror_guards() -> None:
    store = InMemoryJobStore()
    with pytest.raises(KeyError):
        store.set_status("nope", "done")
    with pytest.raises(KeyError):
        store.set_progress("nope", 0.5)
    with pytest.raises(KeyError):
        store.set_result("nope", pd.DataFrame(), pd.DataFrame(), "text")
    with pytest.raises(KeyError):
        store.subscribe("nope")
    with pytest.raises(KeyError):
        store.delete_dataset("nope")
    with pytest.raises(KeyError):
        store.set_error("nope", "boom")


def test_store_set_error_emits_event() -> None:
    store = InMemoryJobStore()
    store.create("j", object())
    store.set_error("j", "boom")
    job = store.get("j")
    assert job.status == "error"
    assert job.error == "boom"
    assert job.finished is not None


def test_store_emit_unknown_job_is_noop() -> None:
    store = InMemoryJobStore()
    # _emit on a job that doesn't exist must not raise.
    store._emit("missing", {"type": "progress"})


@pytest.mark.asyncio
async def test_store_emit_uses_call_soon_threadsafe() -> None:
    """When created inside a running loop, events are scheduled via the loop."""
    store = InMemoryJobStore()
    store.create("j", object())  # running loop captured
    store.set_progress("j", 0.5, message="half")
    queue = store.subscribe("j")
    event = await asyncio.wait_for(queue.get(), timeout=2.0)
    assert event["type"] == "progress"
    assert event["progress"] == 0.5
    assert event["message"] == "half"


# --------------------------------------------------------------------------- #
# workers.py — category filter, parallel executor, error swallowing
# --------------------------------------------------------------------------- #
def test_select_categories_filters_and_raises_on_empty() -> None:
    wordlists = merge_wordlists(
        [pd.io.common.StringIO(WORDLIST_CSV)],
        prefixes={"wl.csv": "wl1"},
        names=["wl.csv"],
    )
    cat = wordlists[0].category_names[0]  # 'wl1_Affect'
    selected = _select_categories(wordlists, [cat])
    assert selected and cat in selected[0].category_names
    with pytest.raises(KeyError):
        _select_categories(wordlists, ["Nonexistent"])


def test_counting_executor_parallel_branch() -> None:
    executor, owns = _counting_executor(2)
    try:
        assert executor is not None and owns is True
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    # Serial branch.
    executor2, owns2 = _counting_executor(None)
    assert executor2 is None and owns2 is False


def test_records_to_native_empty_returns_empty() -> None:
    assert _records_to_native([]) == []


def _make_store_with_data() -> tuple[InMemoryJobStore, str, list[str]]:
    store = InMemoryJobStore()
    df = read_dataset(pd.io.common.StringIO(DATASET_CSV), name="data.csv")
    store.put_dataset("ds", "data.csv", df)
    wordlists = merge_wordlists(
        [pd.io.common.StringIO(WORDLIST_CSV)],
        prefixes={"wl.csv": "wl1"},
        names=["wl.csv"],
    )
    for wl in wordlists:
        store.put_wordlist(wl.namespace, wl)
    return store, "ds", [w.namespace for w in wordlists]


def test_run_analysis_job_with_categories_and_parallel() -> None:
    store, ds_id, wl_ids = _make_store_with_data()
    req = AnalysisRequest(
        dataset_id=ds_id,
        wordlist_ids=wl_ids,
        text_column="text",
        categories=["wl1_Affect"],
        max_n=3,
        parallel_workers=2,
    )
    store.create("job1", req)
    run_analysis_job("job1", req, store)
    job = store.get("job1")
    assert job.status == "done"
    assert job.enhanced_df is not None
    assert "wl1_Affect_word_count" in job.enhanced_df.columns


def test_run_analysis_job_missing_text_column_sets_error() -> None:
    """Bypassing route validation, the worker raises NoTextColumnError → error."""
    store, ds_id, wl_ids = _make_store_with_data()
    req = AnalysisRequest(
        dataset_id=ds_id,
        wordlist_ids=wl_ids,
        text_column="nope",
        max_n=3,
    )
    store.create("job2", req)
    run_analysis_job("job2", req, store)
    job = store.get("job2")
    assert job.status == "error"
    assert "nope" in (job.error or "")


def test_run_analysis_job_missing_dataset_sets_error() -> None:
    store = InMemoryJobStore()
    req = AnalysisRequest(
        dataset_id="missing",
        wordlist_ids=["wl1"],
        text_column="text",
        max_n=3,
    )
    store.create("job3", req)
    run_analysis_job("job3", req, store)
    job = store.get("job3")
    assert job.status == "error"
    assert job.error is not None


# --------------------------------------------------------------------------- #
# analysis.py WS — unknown job (4404), error terminal, client disconnect w/ backlog
# --------------------------------------------------------------------------- #
def test_ws_unknown_job_closes_4404(store: InMemoryJobStore) -> None:
    settings = Settings(max_n_default=3, default_parallel_workers=1)
    app = app_module.create_app(settings=settings)
    app.dependency_overrides[get_job_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_job_runner] = SyncExecutor
    with pytest.raises(Exception):  # noqa: B017, SIM117 - WS close surfaces here
        with TestClient(app) as tc, tc.websocket_connect("/analysis/nope/events") as ws:
            ws.receive_json()


def test_ws_emits_error_when_job_already_errored(store: InMemoryJobStore) -> None:
    settings = Settings(max_n_default=3, default_parallel_workers=1)
    app = app_module.create_app(settings=settings)
    app.dependency_overrides[get_job_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_job_runner] = SyncExecutor
    store.create("errjob", object())
    store.set_error("errjob", "kaboom")
    with TestClient(app) as tc, tc.websocket_connect("/analysis/errjob/events") as ws:
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["message"] == "kaboom"


def test_ws_client_disconnect_drains_backlog(store: InMemoryJobStore) -> None:
    """Client disconnects mid-stream with events still queued → finally drain."""
    settings = Settings(max_n_default=3, default_parallel_workers=1)
    app = app_module.create_app(settings=settings)
    app.dependency_overrides[get_job_store] = lambda: store
    app.dependency_overrides[get_settings] = lambda: settings

    released = threading.Event()
    runner = ThreadPoolExecutor(max_workers=1)
    app.dependency_overrides[get_job_runner] = lambda: runner

    def slow_job(job_id: str, request: object, s: object) -> None:
        s.set_progress(job_id, 0.5, message="one")  # type: ignore[attr-defined]
        s.set_progress(job_id, 0.7, message="two")  # type: ignore[attr-defined]
        released.wait(timeout=5)
        s.set_status(job_id, "done")  # type: ignore[attr-defined]

    original = analysis_route.run_analysis_job
    analysis_route.run_analysis_job = slow_job  # type: ignore[assignment]
    try:
        with TestClient(app) as tc:
            ds = tc.post(
                "/datasets",
                files={"file": ("data.csv", DATASET_CSV.encode(), "text/csv")},
            ).json()
            wl = tc.post(
                "/wordlists",
                files=[("files", ("wl.csv", WORDLIST_CSV.encode(), "text/csv"))],
                data={"prefixes": "wl1"},
            ).json()
            job = tc.post(
                "/analysis",
                json={
                    "dataset_id": ds["id"],
                    "wordlist_ids": wl["ids"],
                    "text_column": "text",
                    "max_n": 3,
                },
            ).json()
            with tc.websocket_connect(f"/analysis/{job['job_id']}/events") as ws:
                first = ws.receive_json()
                assert first["type"] == "progress"
                # Close from the client side while the job is still producing.
            # Release the job so it finishes + pushes 'done' into the queue that
            # the (now-closed) consumer drains in its `finally` block.
            released.set()
    finally:
        analysis_route.run_analysis_job = original  # type: ignore[assignment]
        runner.shutdown(wait=True)


# --------------------------------------------------------------------------- #
# errors.py — empty ValueError fallback + RequestValidationError handler
# --------------------------------------------------------------------------- #
def test_empty_valueerror_uses_fallback_detail() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise ValueError()

    with TestClient(app, raise_server_exceptions=False) as tc:
        r = tc.get("/boom")
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "invalid_argument"
        assert body["detail"] == "The request arguments are invalid."


def test_request_validation_handler() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.post("/thing")
    def thing(payload: dict[str, Any]) -> dict[str, Any]:
        return payload

    with TestClient(app) as tc:
        # Malformed JSON body triggers RequestValidationError.
        r = tc.post("/thing", content=b"not-json", headers={"content-type": "application/json"})
        assert r.status_code == 422
        body = r.json()
        assert body["code"] == "validation_error"
        assert "errors" in body


# --------------------------------------------------------------------------- #
# deps.py — the lru_cached singletons (tests override them, so call the real ones)
# --------------------------------------------------------------------------- #
def test_deps_singletons() -> None:
    get_settings.cache_clear()
    get_job_store.cache_clear()
    get_job_runner.cache_clear()
    s1 = get_settings()
    assert s1 is get_settings()
    st1 = get_job_store()
    assert st1 is get_job_store()
    runner = get_job_runner()
    try:
        assert isinstance(runner, type(get_job_runner()))  # same type each call
    finally:
        runner.shutdown(wait=True)
        get_settings.cache_clear()
        get_job_store.cache_clear()
        get_job_runner.cache_clear()


# --------------------------------------------------------------------------- #
# app.py — run() entrypoint: --generate-openapi + uvicorn.run
# --------------------------------------------------------------------------- #
def test_run_generate_openapi(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    app_module.run(["--generate-openapi"])
    spec = (tmp_path / "api" / "openapi.json").read_text(encoding="utf-8")
    assert '"openapi"' in spec
    assert "/analysis" in spec


def test_run_invokes_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    app_module.run([])
    assert captured["app"] == "wordcount.api.app:create_app"
    assert captured["factory"] is True
    assert captured["port"] == 8000


def test_run_reload_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    app_module.run(["--reload"])
    assert captured["reload"] is True


# --------------------------------------------------------------------------- #
# "not ready" branches: a created-but-not-run job has no result → 404.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_endpoints_return_404_when_result_not_ready(
    client: Any, store: InMemoryJobStore
) -> None:
    store.create("pending", object())
    # enhanced
    assert (await client.get("/analysis/pending/enhanced/columns")).status_code == 404
    assert (await client.get("/analysis/pending/enhanced")).status_code == 404
    # detected
    assert (
        await client.get("/analysis/pending/detected/wl1_Affect", params={"top_n": 5})
    ).status_code == 404
    # export
    assert (await client.get("/export/pending", params={"format": "csv"})).status_code == 404
    # stats
    assert (
        await client.post("/stats/pearson", json={"job_id": "pending", "col1": "a", "col2": "b"})
    ).status_code == 404


@pytest.mark.asyncio
async def test_export_parquet_importerror_returns_501(
    client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the ``parquet`` extra is absent the export route surfaces 501."""

    def _raise(_: object) -> bytes:
        raise ImportError("pyarrow is not installed")

    monkeypatch.setattr(export_route, "to_parquet_bytes", _raise)
    ds = await upload_dataset(client, DATASET_CSV)
    wl = await upload_wordlist(client, WORDLIST_CSV, prefix="wl1")
    job = await run_analysis(client, ds["id"], wl["ids"])
    resp = await client.get(f"/export/{job['job_id']}", params={"format": "parquet"})
    assert resp.status_code == 501
    assert "pyarrow" in resp.json()["detail"]
