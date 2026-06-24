"""Shared fixtures for the FastAPI API tests.

The store + runner are overridden per-test via ``app.dependency_overrides`` (the
FastAPI composability idiom) so no state leaks between tests. A
:class:`SyncExecutor` runs analysis jobs synchronously inside the request thread,
making most tests deterministic without polling. The WebSocket live-progress
test swaps in a real background executor + a threading.Event handshake.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from concurrent.futures import Future

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from wordcount.api.app import create_app
from wordcount.api.deps import get_job_runner, get_job_store, get_settings
from wordcount.api.settings import Settings
from wordcount.api.storage import InMemoryJobStore


class SyncExecutor:
    """Run ``submit`` synchronously and return a completed Future.

    Makes POST /analysis finish the job before returning 202, so tests need no
    polling. Not a real ``ThreadPoolExecutor`` subclass, but duck-typed.
    """

    def submit(
        self, fn: Callable[..., object], /, *args: object, **kwargs: object
    ) -> Future[object]:
        fut: Future[object] = Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - mirror real executor semantics
            fut.set_exception(exc)
        return fut

    def shutdown(self, wait: bool = True) -> None:  # noqa: ARG002
        pass


@pytest.fixture
def store() -> InMemoryJobStore:
    return InMemoryJobStore()


@pytest.fixture
def settings() -> Settings:
    return Settings(max_n_default=3, default_parallel_workers=1, cors_origins=["*"])


@pytest.fixture
def app(store: InMemoryJobStore, settings: Settings) -> object:
    application = create_app(settings=settings)
    application.dependency_overrides[get_job_store] = lambda: store
    application.dependency_overrides[get_settings] = lambda: settings
    application.dependency_overrides[get_job_runner] = SyncExecutor
    return application


@pytest_asyncio.fixture
async def client(app: object) -> AsyncIterator[httpx.AsyncClient]:
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# --------------------------------------------------------------------------- #
# Sample dataset + wordlist content
# --------------------------------------------------------------------------- #
DATASET_CSV = (
    "id,text,group,score\n"
    "1,I am happy and glad,A,5\n"
    "2,I am sad and angry,B,2\n"
    "3,happy happy joy,A,4\n"
    "4,sad mad furious,B,1\n"
    "5,glad glad happy,A,5\n"
    "6,furious mad angry,B,1\n"
)

#: 'happy' (exact) + 'happ*' (wildcard→prefix happ) both match 'happy'/'happier':
#: double-counted per the preserved legacy convention.
WORDLIST_CSV = "DicTerm,Affect\nhappy,X\nglad,X\nsad,X\nangry,X\nhapp*,X\n"


async def upload_dataset(client: httpx.AsyncClient, content: str, name: str = "data.csv") -> dict:
    """POST /datasets helper; returns the parsed JSON body."""
    resp = await client.post(
        "/datasets",
        files={"file": (name, content.encode(), "text/csv")},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def upload_wordlist(
    client: httpx.AsyncClient, content: str, prefix: str, name: str = "wl.csv"
) -> dict:
    """POST /wordlists helper for a single file with a prefix (namespace)."""
    resp = await client.post(
        "/wordlists",
        files=[("files", (name, content.encode(), "text/csv"))],
        data={"prefixes": prefix},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


async def run_analysis(
    client: httpx.AsyncClient,
    dataset_id: str,
    wordlist_ids: list[str],
    *,
    text_column: str = "text",
    categories: list[str] | None = None,
    max_n: int = 3,
) -> dict:
    """POST /analysis helper; returns the job body (already done under SyncExecutor)."""
    body: dict = {
        "dataset_id": dataset_id,
        "wordlist_ids": wordlist_ids,
        "text_column": text_column,
        "max_n": max_n,
    }
    if categories is not None:
        body["categories"] = categories
    resp = await client.post("/analysis", json=body)
    assert resp.status_code == 202, resp.text
    job = resp.json()
    # Under SyncExecutor the job is already done; confirm via the status route.
    status = await client.get(f"/analysis/{job['job_id']}")
    assert status.status_code == 200, status.text
    return job
