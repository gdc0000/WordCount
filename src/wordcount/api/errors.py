"""RFC 9457 ProblemDetails mapping for core exceptions.

``core/`` raises typed :class:`~wordcount.core.models.WordcountError` subclasses
(each with a stable ``code``); the API translates them into JSON ProblemDetails
with the right HTTP status and a friendly ``detail``. Raw tracebacks are logged
via :mod:`logging`, never leaked (fixes §6.3 — no more ``st.error(exc)``). Each
core exception maps to a stable error ``code`` so the UX layer can branch on it.

ProblemDetails body shape (RFC 9457)::

    {
      "type": "https://wordcount.dev/errors/unsupported_format",
      "title": "Unsupported format",
      "status": 415,
      "detail": "Unsupported dataset format '.foo': ...",
      "code": "unsupported_format"
    }
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.requests import Request

from wordcount.core.models import (
    AnalysisConfigError,
    MissingDicTermColumnError,
    NoCategoryColumnsError,
    NoTextColumnError,
    UnsupportedFormatError,
    WordcountError,
)

logger = logging.getLogger("wordcount.api")

#: Base URI for problem ``type`` fields.
_TYPE_BASE = "https://wordcount.dev/errors/"

#: Per-error status + title. Core ``code`` is the leaf of the ``type`` URI.
_STATUS_MAP: dict[type[Exception], tuple[int, str]] = {
    UnsupportedFormatError: (415, "Unsupported format"),
    MissingDicTermColumnError: (422, "Missing 'DicTerm' column"),
    NoCategoryColumnsError: (422, "No category columns"),
    NoTextColumnError: (422, "No text column"),
    AnalysisConfigError: (422, "Invalid analysis configuration"),
    WordcountError: (400, "WordCount error"),
}


def problem_response(
    exc: Exception,
    status: int,
    title: str,
    code: str | None = None,
    detail: str | None = None,
) -> JSONResponse:
    """Build a JSON ProblemDetails response (RFC 9457)."""
    leaf = code or getattr(exc, "code", None) or "error"
    return JSONResponse(
        status_code=status,
        content={
            "type": f"{_TYPE_BASE}{leaf}",
            "title": title,
            "status": status,
            "detail": detail or str(exc) or title,
            "code": leaf,
        },
    )


def _wordcount_problem(exc: WordcountError) -> JSONResponse:
    # Most-specific registered class wins.
    for cls in type(exc).__mro__:
        if cls in _STATUS_MAP:
            status, title = _STATUS_MAP[cls]
            return problem_response(exc, status, title, code=exc.code)
    # Fallback (shouldn't happen — base WordcountError is registered).
    return problem_response(exc, 400, "WordCount error", code=exc.code)  # pragma: no cover


def _not_found(exc: KeyError) -> JSONResponse:
    return problem_response(
        exc,
        404,
        "Resource not found",
        code="not_found",
        detail=f"Resource not found: {exc}",
    )


def _validation_problem(exc: ValueError) -> JSONResponse:
    return problem_response(
        exc,
        422,
        "Unprocessable request",
        code="invalid_argument",
        detail=str(exc) or "The request arguments are invalid.",
    )


def _server_error(exc: Exception) -> JSONResponse:
    # Log the full traceback for operators; never leak it to the client (§6.3).
    logger.exception("Unhandled error in API")
    return problem_response(
        exc,
        500,
        "Internal server error",
        code="internal_error",
        detail="An internal error occurred. It has been logged.",
    )


def _request_validation_problem(exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={
            "type": f"{_TYPE_BASE}validation_error",
            "title": "Request validation failed",
            "status": 422,
            "detail": "One or more request fields failed validation.",
            "code": "validation_error",
            "errors": jsonable_encoder(exc.errors()),
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all ProblemDetails handlers on a FastAPI app."""

    @app.exception_handler(WordcountError)
    async def _handle_wordcount(_: Request, exc: WordcountError) -> JSONResponse:
        return _wordcount_problem(exc)

    @app.exception_handler(KeyError)
    async def _handle_keyerror(_: Request, exc: KeyError) -> JSONResponse:
        return _not_found(exc)

    @app.exception_handler(ValueError)
    async def _handle_valueerror(_: Request, exc: ValueError) -> JSONResponse:
        return _validation_problem(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        return _request_validation_problem(exc)

    @app.exception_handler(Exception)
    async def _handle_generic(_: Request, exc: Exception) -> JSONResponse:
        return _server_error(exc)


__all__: list[Any] = ["problem_response", "register_exception_handlers"]
