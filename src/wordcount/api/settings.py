"""API settings (pydantic-settings).

Reads from env vars prefixed ``WORDCOUNT_`` (and an optional ``.env``). Injected
via :func:`api.deps.get_settings` so tests override per-test via
``app.dependency_overrides`` (the FastAPI composability idiom).

These are *connector* concerns (default ``max_n``, worker counts, CORS, upload
limits), **not** business concerns — the pure ``core/`` takes ``max_n`` as a
plain parameter (fixes §5.1: the default no longer lives in the business code).
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Ephemeral connector configuration. Stateless across requests."""

    model_config = SettingsConfigDict(
        env_prefix="WORDCOUNT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: Default n-gram order for analysis (the ``max_n`` parameter to core).
    max_n_default: int = 3
    #: Default parallel workers for the per-document counting executor.
    default_parallel_workers: int = 4
    #: Ephemeral storage backend (``memory`` now; ``redis`` is a drop-in later).
    storage_backend: str = "memory"
    #: CORS allowed origins — any UX origin may call the API.
    cors_origins: list[str] = ["*"]
    #: Max upload size in MB for datasets/wordlists.
    max_upload_mb: int = 200
    #: API title shown in /docs.
    api_title: str = "WordCount Statistics API"
    #: API version surfaced at /healthz.
    api_version: str = "0.1.0"


__all__ = ["Settings"]
