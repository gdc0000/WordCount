"""Ephemeral job/resource store: a swappable, Redis-shaped dependency.

The API keeps no per-user session (stateless, §6.4). The only server state is
an ephemeral store for in-flight analyses plus the uploaded datasets/wordlists
they reference. ``JobStore`` is a :class:`Protocol` so a Redis-backed impl is a
drop-in for horizontal scale later — the audit's "scale" concern made explicit.

``InMemoryJobStore`` is the single-process impl: a dict of jobs/datasets/
wordlists plus a per-job ``asyncio.Queue`` for WebSocket fan-out (§5.4 + §6.4).

Threading note: the analysis worker runs in a thread (see :mod:`api.workers`),
while the queue + WebSocket consumer run on the event loop. The store captures
the running loop at :meth:`create` time and pushes progress events with
``loop.call_soon_threadsafe`` so cross-thread ``put_nowait`` is safe. When no
loop is running (sync unit tests), it falls back to a direct ``put_nowait``.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import pandas as pd

from wordcount.core.models import Wordlist

JobStatus = Literal["pending", "running", "done", "error"]


@dataclass
class DatasetRecord:
    """An uploaded dataset held in the ephemeral store."""

    id: str
    name: str
    df: pd.DataFrame


@dataclass
class WordlistRecord:
    """One namespaced wordlist held in the ephemeral store."""

    id: str
    wordlist: Wordlist


@dataclass
class Job:
    """An analysis job's mutable state."""

    job_id: str
    request: Any  # AnalysisRequest; typed Any to avoid an api<->schema cycle
    status: JobStatus = "pending"
    progress: float = 0.0
    started: float = field(default_factory=time.time)
    finished: float | None = None
    error: str | None = None
    text_column: str | None = None
    enhanced_df: pd.DataFrame | None = None
    analysis_df: pd.DataFrame | None = None
    queue: asyncio.Queue[dict[str, Any]] = field(default_factory=asyncio.Queue)


@runtime_checkable
class JobStore(Protocol):
    """The ephemeral store surface the API depends on (swappable)."""

    # Jobs ------------------------------------------------------------------
    def create(self, job_id: str, request: Any) -> None: ...
    def get(self, job_id: str) -> Job: ...
    def set_status(self, job_id: str, status: JobStatus) -> None: ...
    def set_progress(self, job_id: str, progress: float, message: str | None = None) -> None: ...
    def set_result(
        self,
        job_id: str,
        enhanced_df: pd.DataFrame,
        analysis_df: pd.DataFrame,
        text_column: str | None,
    ) -> None: ...
    def set_error(self, job_id: str, error: str) -> None: ...
    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]: ...

    # Datasets --------------------------------------------------------------
    def put_dataset(self, dataset_id: str, name: str, df: pd.DataFrame) -> None: ...
    def get_dataset(self, dataset_id: str) -> DatasetRecord: ...
    def delete_dataset(self, dataset_id: str) -> None: ...

    # Wordlists -------------------------------------------------------------
    def put_wordlist(self, wordlist_id: str, wordlist: Wordlist) -> None: ...
    def get_wordlists(self, wordlist_ids: list[str]) -> list[Wordlist]: ...


class InMemoryJobStore:
    """Single-process dict-backed implementation of :class:`JobStore`.

    Thread-safe enough for one uvicorn process: a lock guards the dicts; the
    per-job queue is pushed to via the captured event loop
    (``call_soon_threadsafe``) so the worker thread never touches asyncio
    directly.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._datasets: dict[str, DatasetRecord] = {}
        self._wordlists: dict[str, WordlistRecord] = {}
        self._loops: dict[str, asyncio.AbstractEventLoop | None] = {}
        self._lock = threading.Lock()

    # Jobs ------------------------------------------------------------------
    def create(self, job_id: str, request: Any) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._lock:
            self._jobs[job_id] = Job(job_id=job_id, request=request)
            self._loops[job_id] = loop

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id!r}")
        return job

    def set_status(self, job_id: str, status: JobStatus) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"job {job_id!r}")
            job.status = status
            if status in ("done", "error"):
                job.finished = time.time()
                job.progress = 1.0 if status == "done" else job.progress
        self._emit(job_id, {"type": status})

    def set_progress(self, job_id: str, progress: float, message: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"job {job_id!r}")
            job.status = "running"
            job.progress = progress
        event: dict[str, Any] = {"type": "progress", "progress": progress}
        if message:
            event["message"] = message
        self._emit(job_id, event)

    def set_result(
        self,
        job_id: str,
        enhanced_df: pd.DataFrame,
        analysis_df: pd.DataFrame,
        text_column: str | None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"job {job_id!r}")
            job.enhanced_df = enhanced_df
            job.analysis_df = analysis_df
            job.text_column = text_column
            job.progress = 1.0

    def set_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(f"job {job_id!r}")
            job.status = "error"
            job.error = error
            job.finished = time.time()
        self._emit(job_id, {"type": "error", "message": error})

    def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"job {job_id!r}")
        return job.queue

    def _emit(self, job_id: str, event: dict[str, Any]) -> None:
        """Push an event to the job's queue, thread-safe across the loop."""
        with self._lock:
            job = self._jobs.get(job_id)
            loop = self._loops.get(job_id)
        if job is None:
            return
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(job.queue.put_nowait, event)
        else:
            # Same-thread or no loop: direct put (unbounded queue, so safe).
            with contextlib.suppress(asyncio.QueueFull):  # pragma: no cover - unbounded
                job.queue.put_nowait(event)

    # Datasets --------------------------------------------------------------
    def put_dataset(self, dataset_id: str, name: str, df: pd.DataFrame) -> None:
        with self._lock:
            self._datasets[dataset_id] = DatasetRecord(id=dataset_id, name=name, df=df)

    def get_dataset(self, dataset_id: str) -> DatasetRecord:
        with self._lock:
            rec = self._datasets.get(dataset_id)
        if rec is None:
            raise KeyError(f"dataset {dataset_id!r}")
        return rec

    def delete_dataset(self, dataset_id: str) -> None:
        with self._lock:
            if self._datasets.pop(dataset_id, None) is None:
                raise KeyError(f"dataset {dataset_id!r}")

    # Wordlists -------------------------------------------------------------
    def put_wordlist(self, wordlist_id: str, wordlist: Wordlist) -> None:
        with self._lock:
            self._wordlists[wordlist_id] = WordlistRecord(id=wordlist_id, wordlist=wordlist)

    def get_wordlists(self, wordlist_ids: list[str]) -> list[Wordlist]:
        result: list[Wordlist] = []
        with self._lock:
            for wid in wordlist_ids:
                rec = self._wordlists.get(wid)
                if rec is None:
                    raise KeyError(f"wordlist {wid!r}")
                result.append(rec.wordlist)
        return result


__all__ = [
    "DatasetRecord",
    "InMemoryJobStore",
    "Job",
    "JobStatus",
    "JobStore",
    "WordlistRecord",
]
