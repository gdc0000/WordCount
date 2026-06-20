"""Smoke test: the package imports and exposes a version.

This is intentionally tiny — it exists so the test suite is green (pytest exits 0)
before real tests arrive in later phases, and so coverage has data to report.
"""

from __future__ import annotations

import wordcount


def test_package_exposes_version() -> None:
    assert isinstance(wordcount.__version__, str)
    assert wordcount.__version__
