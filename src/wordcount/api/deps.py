"""Dependency injection for the API (FastAPI composability idiom).

``get_settings`` / ``get_job_store`` / ``get_job_runner`` are the seams tests
override via ``app.dependency_overrides`` — no globals are mutated directly, so
each test gets an isolated store/runner/settings (the FastAPI way, vs. the
legacy module-global state the audit flagged in §4/§6.4).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from wordcount.api.settings import Settings
from wordcount.api.storage import InMemoryJobStore, JobStore


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton :class:`Settings` (env-driven)."""
    return Settings()


@lru_cache(maxsize=1)
def get_job_store() -> JobStore:
    """Return the singleton ephemeral :class:`JobStore`.

    Tests override this with a fresh ``InMemoryJobStore`` per test via
    ``app.dependency_overrides``.
    """
    return InMemoryJobStore()


@lru_cache(maxsize=1)
def get_job_runner() -> ThreadPoolExecutor:
    """Return a cached executor that runs analysis jobs in background threads.

    The per-document *counting* parallelism is a separate executor created
    inside the worker (see :mod:`api.workers`); this one only offloads the
    whole job from the request thread so ``POST /analysis`` returns 202 fast.
    """
    settings = get_settings()
    return ThreadPoolExecutor(max_workers=max(1, settings.default_parallel_workers))


__all__ = ["get_job_runner", "get_job_store", "get_settings"]
