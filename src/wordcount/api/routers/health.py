"""GET /healthz — liveness + connector config."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from wordcount.api.deps import get_settings
from wordcount.api.schemas import Health
from wordcount.api.settings import Settings

router = APIRouter(tags=["health"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


@router.get("/healthz", response_model=Health)
def health(settings: SettingsDep) -> Health:
    """Liveness probe; surfaces connector config (max_n, parallel workers)."""
    return Health(
        version=settings.api_version,
        max_n=settings.max_n_default,
        parallel=settings.default_parallel_workers,
    )
