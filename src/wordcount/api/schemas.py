"""Pydantic request/response DTOs (wire contracts over core dataclasses).

The API **never accepts or returns a raw DataFrame**; it always goes through a
schema, so the wire contract is stable and the UX layer can be generated from
OpenAPI (``/docs``). These mirror the core dataclasses in
:mod:`wordcount.core.models` but are plain Pydantic models used only for HTTP
serialization — the business logic stays in pure ``core/``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #
class DatasetUpload(BaseModel):
    """Result of POST /datasets."""

    id: str
    name: str
    n_rows: int
    columns: list[str]
    dtype_guesses: dict[str, str]
    preview: list[dict[str, Any]] = Field(default_factory=list)


class DatasetMeta(BaseModel):
    """GET /datasets/{id}."""

    id: str
    name: str
    n_rows: int
    columns: list[str]
    dtype_guesses: dict[str, str]
    preview: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Wordlists
# --------------------------------------------------------------------------- #
class WordlistCategorySummary(BaseModel):
    """One category's term count + samples (replaces generate_summary_list)."""

    category: str
    n_terms: int
    sample_terms: list[str] = Field(default_factory=list)


class WordlistSummary(BaseModel):
    """GET /wordlists/{id}/summary."""

    namespace: str
    categories: list[WordlistCategorySummary]


class WordlistUpload(BaseModel):
    """Result of POST /wordlists."""

    ids: list[str]
    namespaces: list[str]
    categories_per_list: list[int]
    summary: list[WordlistSummary]


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
class AnalysisRequest(BaseModel):
    """POST /analysis body. The default max_n lives here, not in core (§5.1)."""

    dataset_id: str
    wordlist_ids: list[str]
    text_column: str
    categories: list[str] | None = None  # None/empty => all categories
    max_n: int = 3
    parallel_workers: int | None = None


class AnalysisJob(BaseModel):
    """Result of POST /analysis (202)."""

    job_id: str
    status: str = "pending"


class JobStatusView(BaseModel):
    """GET /analysis/{job_id}."""

    job_id: str
    status: str
    progress: float
    started: float
    finished: float | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# Enhanced dataset
# --------------------------------------------------------------------------- #
class EnhancedColumns(BaseModel):
    """GET /analysis/{job_id}/enhanced/columns."""

    columns: list[str]


class EnhancedRecords(BaseModel):
    """GET /analysis/{job_id}/enhanced (paginated)."""

    offset: int
    limit: int | None
    total: int
    columns: list[str]
    records: list[dict[str, Any]]


# --------------------------------------------------------------------------- #
# Detected terms
# --------------------------------------------------------------------------- #
class TermFreq(BaseModel):
    """One detected term with its frequency and doc-frequency (§2.2)."""

    term: str
    frequency: int
    doc_frequency: int


class DetectedTerms(BaseModel):
    """GET /analysis/{job_id}/detected/{category}?top_n=."""

    category: str
    top_n: int
    terms: list[TermFreq]


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #
class PearsonRequest(BaseModel):
    job_id: str
    col1: str
    col2: str


class PearsonResponse(BaseModel):
    col1: str
    col2: str
    coefficient: float
    p_value: float
    n: int


class GroupStatsDTO(BaseModel):
    category: str
    n: int
    mean: float
    sem: float
    ci_lower: float
    ci_upper: float


class AnovaRequest(BaseModel):
    job_id: str
    cat_var: str
    num_var: str


class AnovaResponse(BaseModel):
    cat_var: str
    num_var: str
    p_value: float
    significant: bool
    group_stats: list[GroupStatsDTO]
    tukey_rows: list[dict[str, Any]]
    table: list[dict[str, Any]]


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #
class Health(BaseModel):
    """GET /healthz — liveness + connector config."""

    status: str = "ok"
    version: str
    max_n: int
    parallel: int


__all__ = [
    "AnalysisJob",
    "AnalysisRequest",
    "AnovaRequest",
    "AnovaResponse",
    "DatasetMeta",
    "DatasetUpload",
    "DetectedTerms",
    "EnhancedColumns",
    "EnhancedRecords",
    "GroupStatsDTO",
    "Health",
    "JobStatusView",
    "PearsonRequest",
    "PearsonResponse",
    "TermFreq",
    "WordlistCategorySummary",
    "WordlistSummary",
    "WordlistUpload",
]
