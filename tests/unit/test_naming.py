"""Tests for ``wordcount.core.naming`` — the §10 "model to clone", now with the
``sanitize_columns`` collision regression (§3.2).

Pure: no pandas, no streamlit, no fastapi.
"""

from __future__ import annotations

import pytest

from wordcount.core.naming import sanitize_columns, sanitize_identifier, uniquify


# --------------------------------------------------------------------------- #
# sanitize_identifier
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Affect", "Affect"),
        ("Affect-Modal", "Affect_Modal"),
        ("Affect.Modal!", "Affect_Modal"),
        ("  spaced  out  ", "spaced_out"),
        ("---leading/trailing---", "leading_trailing"),
        ("123abc", "123abc"),
        ("already_clean", "already_clean"),
    ],
)
def test_sanitize_identifier_basic(raw: str, expected: str) -> None:
    assert sanitize_identifier(raw) == expected


def test_sanitize_identifier_collapses_runs() -> None:
    # Multiple consecutive non-identifier chars collapse to a single underscore.
    assert sanitize_identifier("a,,,.;b") == "a_b"


def test_sanitize_identifier_empty_falls_back() -> None:
    assert sanitize_identifier("") == "wordlist"
    assert sanitize_identifier("!!!") == "wordlist"
    assert sanitize_identifier("   ") == "wordlist"
    assert sanitize_identifier("", fallback="col") == "col"


# --------------------------------------------------------------------------- #
# uniquify
# --------------------------------------------------------------------------- #
def test_uniquify_passthrough_when_free() -> None:
    existing: set[str] = {"a", "b"}
    assert uniquify("c", existing) == "c"
    # Does not mutate the input set.
    assert existing == {"a", "b"}


def test_uniquify_appends_suffix() -> None:
    existing = {"a", "a_2", "a_3"}
    assert uniquify("a", existing) == "a_4"


def test_uniquify_first_collision_is_2() -> None:
    assert uniquify("x", {"x"}) == "x_2"


# --------------------------------------------------------------------------- #
# sanitize_columns — the §3.2 regression
# --------------------------------------------------------------------------- #
def test_sanitize_columns_preserves_length_and_uniqueness() -> None:
    cols = ["Affect", "Modal", "Affect-Modal", "n_tokens"]
    result = sanitize_columns(cols)
    assert len(result) == len(cols)
    assert len(set(result)) == len(result)


def test_sanitize_columns_collision_regression() -> None:
    """Two originals differing only in stripped chars must NOT collapse.

    This is the exact §3.2 bug: the old ``str.replace("[^A-Za-z0-9_]", "")``
    turned ``a.b`` and ``a-b`` into the same ``ab``-ish header, producing a
    duplicate column. With :func:`uniquify` they stay distinct.
    """
    result = sanitize_columns(["a.b", "a-b", "a_b"])
    assert len(result) == 3
    assert len(set(result)) == 3
    # The first one keeps the clean form; the others get suffixed.
    assert result[0] == "a_b"


def test_sanitize_columns_handles_non_string_inputs() -> None:
    # Column names can arrive as ints (e.g. unnamed integer columns).
    result = sanitize_columns([1, 2, 2])  # type: ignore[list-item]
    assert result == ["1", "2", "2_2"]


def test_sanitize_columns_empty_input() -> None:
    assert sanitize_columns([]) == []


def test_sanitize_columns_all_empty_strings_get_fallbacks() -> None:
    result = sanitize_columns(["", "", ""])
    assert len(result) == 3
    assert len(set(result)) == 3
    assert result[0] == "wordlist"
