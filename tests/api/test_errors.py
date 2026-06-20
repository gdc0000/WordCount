"""Error mapping: core exceptions → RFC 9457 ProblemDetails."""

from __future__ import annotations

import json
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from wordcount.api.errors import problem_response, register_exception_handlers
from wordcount.core.models import (
    MissingDicTermColumnError,
    NoCategoryColumnsError,
    NoTextColumnError,
    UnsupportedFormatError,
    WordcountError,
)


def test_problem_response_shape() -> None:
    resp = problem_response(
        UnsupportedFormatError("bad '.foo'"), 415, "Unsupported format", code="unsupported_format"
    )
    assert resp.status_code == 415
    body = resp.body  # JSONResponse renders bytes
    data: Any = json.loads(body)
    assert data["type"] == "https://wordcount.dev/errors/unsupported_format"
    assert data["status"] == 415
    assert data["code"] == "unsupported_format"
    assert "bad" in data["detail"]


def _app_with_routes() -> FastAPI:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/raise/{kind}")
    def raise_(kind: str) -> None:
        if kind == "unsupported":
            raise UnsupportedFormatError("nope")
        if kind == "missing_dicterm":
            raise MissingDicTermColumnError("nope")
        if kind == "no_categories":
            raise NoCategoryColumnsError("nope")
        if kind == "no_text":
            raise NoTextColumnError("nope")
        if kind == "wordcount":
            raise WordcountError("boom")
        if kind == "key":
            raise KeyError("thing")
        if kind == "value":
            raise ValueError("bad arg")
        if kind == "generic":
            raise RuntimeError("kaboom")

    return app


def test_unsupported_format_maps_415() -> None:
    with TestClient(_app_with_routes()) as tc:
        r = tc.get("/raise/unsupported")
        assert r.status_code == 415
        assert r.json()["code"] == "unsupported_format"


def test_missing_dicterm_maps_422() -> None:
    with TestClient(_app_with_routes()) as tc:
        assert tc.get("/raise/missing_dicterm").status_code == 422


def test_no_categories_maps_422() -> None:
    with TestClient(_app_with_routes()) as tc:
        assert tc.get("/raise/no_categories").status_code == 422


def test_no_text_maps_422() -> None:
    with TestClient(_app_with_routes()) as tc:
        assert tc.get("/raise/no_text").status_code == 422


def test_wordcount_base_maps_400() -> None:
    with TestClient(_app_with_routes()) as tc:
        r = tc.get("/raise/wordcount")
        assert r.status_code == 400
        assert r.json()["code"] == "wordcount_error"


def test_keyerror_maps_404() -> None:
    with TestClient(_app_with_routes()) as tc:
        r = tc.get("/raise/key")
        assert r.status_code == 404
        assert r.json()["code"] == "not_found"


def test_valueerror_maps_422() -> None:
    with TestClient(_app_with_routes()) as tc:
        r = tc.get("/raise/value")
        assert r.status_code == 422
        assert r.json()["code"] == "invalid_argument"


def test_generic_maps_500_no_traceback_leak() -> None:
    with TestClient(_app_with_routes(), raise_server_exceptions=False) as tc:
        r = tc.get("/raise/generic")
        assert r.status_code == 500
        body = r.json()
        assert body["code"] == "internal_error"
        # The raw traceback / message must NOT leak to the client (§6.3).
        assert "kaboom" not in body["detail"]
