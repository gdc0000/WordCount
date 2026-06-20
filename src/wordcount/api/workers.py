"""Background analysis worker: runs a job in a thread, reporting progress.

``run_analysis_job`` is submitted to the job-runner executor by the analysis
route. It is the *only* place that wires the pure ``core/`` pipeline together
for the API: load dataset + wordlists from the store → build config → count →
enhance → store result. No business logic lives here; this is orchestration of
core functions plus progress→store writes (§5.4: the core's ``progress``
callback is fed a job-store writer closure).

Runs in a worker thread, so it never touches asyncio directly — progress events
are pushed to the job's queue via the store's ``call_soon_threadsafe`` bridge
(see :mod:`api.storage`). Errors are caught, logged, and surfaced as a job
``error`` status + ``error`` event (never a raised exception in the request
thread).
"""

from __future__ import annotations

import logging
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

from wordcount.api.storage import JobStore
from wordcount.core.counting import analyze_documents, build_analysis_config
from wordcount.core.enhance import enhance_dataset
from wordcount.core.models import NoTextColumnError, Wordlist

logger = logging.getLogger("wordcount.api.worker")


def _select_categories(wordlists: list[Wordlist], categories: list[str] | None) -> list[Wordlist]:
    """Filter wordlists to the requested category names (None/empty => all)."""
    if not categories:
        return wordlists
    wanted = set(categories)
    result: list[Wordlist] = []
    for wl in wordlists:
        selected = {name: terms for name, terms in wl.categories if name in wanted}
        if selected:
            result.append(Wordlist.from_mapping(wl.namespace, selected))
    if not result:
        raise KeyError("none of the requested categories exist in the wordlists")
    return result


def _counting_executor(parallel_workers: int | None) -> tuple[Executor | None, bool]:
    """Return (executor, owns) for per-document counting parallelism."""
    if parallel_workers and parallel_workers > 1:
        return ThreadPoolExecutor(max_workers=parallel_workers), True
    return None, False


def run_analysis_job(job_id: str, request: Any, store: JobStore) -> None:
    """Execute one analysis job end-to-end, writing progress + result to store.

    Raises are swallowed and converted to a job ``error`` status so the
    background thread never crashes silently. The store's error event lets the
    WebSocket consumer terminate cleanly.
    """
    try:
        store.set_status(job_id, "running")

        dataset = store.get_dataset(request.dataset_id)
        wordlists = store.get_wordlists(list(request.wordlist_ids))
        selected = _select_categories(wordlists, request.categories)

        if request.text_column not in dataset.df.columns:
            raise NoTextColumnError(
                f"Text column {request.text_column!r} not in dataset columns "
                f"{list(dataset.df.columns)}."
            )

        config = build_analysis_config(selected, request.max_n)
        documents = dataset.df[request.text_column].fillna("").astype(str).tolist()

        def _on_progress(done: int, total: int) -> None:
            progress = (done / total) if total else 1.0
            store.set_progress(job_id, progress, message=f"{done}/{total} documents")

        executor, owns = _counting_executor(request.parallel_workers)
        try:
            analysis_df = analyze_documents(
                documents, config, progress=_on_progress, executor=executor
            )
        finally:
            if owns and executor is not None:
                executor.shutdown(wait=True)

        enhanced = enhance_dataset(dataset.df, analysis_df, text_column=request.text_column)
        store.set_result(job_id, enhanced, analysis_df, request.text_column)
        store.set_status(job_id, "done")
    except Exception as exc:  # noqa: BLE001 - background thread must not crash
        logger.exception("Analysis job %s failed", job_id)
        store.set_error(job_id, str(exc))


__all__ = ["run_analysis_job"]
