"""Analysis job endpoints — the async core of the API.

POST /analysis                — start a job (202 + job_id).
GET  /analysis/{job_id}       — poll status/progress.
WS   /analysis/{job_id}/events — live progress/done/error stream.

Long work returns a ``job_id`` immediately and runs in a background thread
(:func:`api.workers.run_analysis_job`). Progress is the *same callback
abstraction* the pure core exposes (§5.4); here it writes to the
:class:`JobStore` and fans out to any WebSocket subscribers. ``job_id`` is the
only server-side key (§6.4 — one source of truth, vs. the legacy dual
session_state + re-inference).
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from wordcount.api.deps import get_job_runner, get_job_store
from wordcount.api.schemas import AnalysisJob, AnalysisRequest, JobStatusView
from wordcount.api.storage import Job, JobStore
from wordcount.api.workers import run_analysis_job
from wordcount.core.models import NoTextColumnError

router = APIRouter(prefix="/analysis", tags=["analysis"])

StoreDep = Annotated[JobStore, Depends(get_job_store)]
RunnerDep = Annotated[ThreadPoolExecutor, Depends(get_job_runner)]


def _status_view(job: Job) -> JobStatusView:
    return JobStatusView(
        job_id=job.job_id,
        status=job.status,
        progress=job.progress,
        started=job.started,
        finished=job.finished,
        error=job.error,
    )


def _validate_request(req: AnalysisRequest, store: JobStore) -> None:
    """Fail fast on bad references before the job is created (sync errors → 4xx)."""
    dataset = store.get_dataset(req.dataset_id)  # KeyError → 404
    store.get_wordlists(req.wordlist_ids)  # KeyError → 404
    if req.text_column not in dataset.df.columns:
        raise NoTextColumnError(
            f"Text column {req.text_column!r} not in dataset columns {list(dataset.df.columns)}."
        )


@router.post("", response_model=AnalysisJob, status_code=202)
def create_analysis(
    req: AnalysisRequest,
    store: StoreDep,
    runner: RunnerDep,
) -> AnalysisJob:
    """Start an analysis job; returns 202 with a job_id immediately."""
    _validate_request(req, store)
    job_id = uuid.uuid4().hex
    store.create(job_id, req)
    runner.submit(run_analysis_job, job_id, req, store)
    return AnalysisJob(job_id=job_id, status="pending")


@router.get("/{job_id}", response_model=JobStatusView)
def get_analysis(
    job_id: str,
    store: StoreDep,
) -> JobStatusView:
    """Poll a job's status/progress."""
    return _status_view(store.get(job_id))


@router.websocket("/{job_id}/events")
async def analysis_events(
    ws: WebSocket,
    job_id: str,
    store: StoreDep,
) -> None:
    """Stream progress/done/error events for a job over WebSocket."""
    try:
        job = store.get(job_id)  # raises KeyError if unknown
    except KeyError:
        await ws.close(code=4404)
        return

    await ws.accept()
    queue = store.subscribe(job_id)

    # If the job already finished before subscription, emit a terminal event so
    # the client isn't stuck waiting on an empty queue.
    if job.status in ("done", "error"):
        event: dict[str, Any] = (
            {"type": "done"} if job.status == "done" else {"type": "error", "message": job.error}
        )
        await ws.send_json(event)
        await ws.close()
        return

    try:
        while True:
            event = await queue.get()
            await ws.send_json(event)
            if event.get("type") in ("done", "error"):
                await ws.close()
                return
    except WebSocketDisconnect:  # pragma: no cover - client race, not reachable via TestClient
        return
    finally:
        # Drain any remaining events so the queue doesn't grow unbounded.
        try:
            while not queue.empty():
                queue.get_nowait()  # pragma: no cover - client race, not reachable via TestClient
        except asyncio.QueueEmpty:  # pragma: no cover - race guard
            pass


__all__ = ["router"]
