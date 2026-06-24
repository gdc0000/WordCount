"""Wordlist upload + summary endpoints.

POST /wordlists            — upload one or more files (+ optional prefixes),
                             merge into namespaced Wordlists, store each.
GET  /wordlists/{id}/summary — per-category term counts + samples (replaces
                             the legacy generate_summary_list).
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, UploadFile

from wordcount.api.deps import get_job_store
from wordcount.api.schemas import (
    WordlistCategorySummary,
    WordlistSummary,
    WordlistUpload,
)
from wordcount.api.storage import JobStore
from wordcount.core.io import merge_wordlists
from wordcount.core.models import CategoryTerms, Wordlist

router = APIRouter(prefix="/wordlists", tags=["wordlists"])

#: Number of sample terms returned per category in the summary.
_SAMPLE_TERMS = 10

StoreDep = Annotated[JobStore, Depends(get_job_store)]


def _sample_terms(terms: CategoryTerms) -> list[str]:
    """A small deterministic sample across the four match-strategy buckets."""
    samples: list[str] = []
    samples.extend(sorted(terms.exact_single))
    samples.extend(sorted(terms.exact_multi))
    samples.extend(f"{p}*" for p in terms.wildcard_single)
    samples.extend(f"{p}*" for p in terms.wildcard_multi)
    return samples[:_SAMPLE_TERMS]


def _summary_for(wordlist: Wordlist) -> WordlistSummary:
    cats = [
        WordlistCategorySummary(
            category=name,
            n_terms=terms.n_terms,
            sample_terms=_sample_terms(terms),
        )
        for name, terms in wordlist.categories
    ]
    return WordlistSummary(namespace=wordlist.namespace, categories=cats)


def _save_uploads(files: Sequence[UploadFile]) -> list[tuple[Path, UploadFile]]:
    """Persist each UploadFile to a temp file keeping its original suffix."""
    saved: list[tuple[Path, UploadFile]] = []
    for upload in files:
        original = upload.filename or "wordlist.csv"
        suffix = Path(original).suffix or ".csv"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="wc_wordlist_")
        os.close(fd)
        path = Path(tmp_path)
        with path.open("wb") as fh:
            while True:
                chunk = upload.file.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        saved.append((path, upload))
    return saved


@router.post("", response_model=WordlistUpload, status_code=201)
async def upload_wordlists(
    store: StoreDep,
    files: Annotated[list[UploadFile], File()],
    prefixes: Annotated[list[str] | None, Form()] = None,
) -> WordlistUpload:
    """Upload one or more wordlists; namespaces come from ``prefixes`` or stems."""
    saved = _save_uploads(files)
    paths = [p for p, _ in saved]
    names = [u.filename or f"wordlist{i}.csv" for i, (_, u) in enumerate(saved)]
    try:
        wordlists = merge_wordlists(paths, prefixes, names=names)
    finally:
        for path, _ in saved:
            path.unlink(missing_ok=True)

    ids: list[str] = []
    for wl in wordlists:
        wid = uuid.uuid4().hex
        store.put_wordlist(wid, wl)
        ids.append(wid)

    return WordlistUpload(
        ids=ids,
        namespaces=[wl.namespace for wl in wordlists],
        categories_per_list=[wl.n_categories for wl in wordlists],
        summary=[_summary_for(wl) for wl in wordlists],
    )


@router.get("/{wordlist_id}/summary", response_model=WordlistSummary)
def wordlist_summary(
    wordlist_id: str,
    store: StoreDep,
) -> WordlistSummary:
    """Per-category term counts + samples for one stored wordlist."""
    wordlists = store.get_wordlists([wordlist_id])
    return _summary_for(wordlists[0])


__all__ = ["router"]
