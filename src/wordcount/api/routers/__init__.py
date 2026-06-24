"""FastAPI routers, one per resource/concern."""

from __future__ import annotations

from wordcount.api.routers import (
    analysis,
    datasets,
    detected,
    enhanced,
    export,
    health,
    stats,
    wordlists,
)

__all__ = [
    "analysis",
    "datasets",
    "detected",
    "enhanced",
    "export",
    "health",
    "stats",
    "wordlists",
]
