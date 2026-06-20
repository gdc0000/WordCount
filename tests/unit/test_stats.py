"""Tests for ``wordcount.core.stats`` — the §2.3/§2.4/§2.5 fixes.

Pure core tests: pandas/scipy/statsmodels build fixtures; no streamlit.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from wordcount.core.stats import anova, group_statistics, pearson


# --------------------------------------------------------------------------- #
# pearson
# --------------------------------------------------------------------------- #
def test_pearson_basic_positive() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    result = pearson(df, "x", "y")
    assert result.col1 == "x"
    assert result.col2 == "y"
    assert result.coefficient == pytest.approx(1.0)
    assert result.p_value == pytest.approx(0.0, abs=1e-12)
    assert result.n == 5


def test_pearson_negative_correlation() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 3, 2, 1]})
    result = pearson(df, "x", "y")
    assert result.coefficient == pytest.approx(-1.0)


def test_pearson_drops_na_pairwise() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, np.nan], "y": [2, 4, 6, 99]})
    result = pearson(df, "x", "y")
    assert result.n == 3
    assert result.coefficient == pytest.approx(1.0)


def test_pearson_same_column_raises() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="distinct"):
        pearson(df, "x", "x")


def test_pearson_too_few_observations() -> None:
    df = pd.DataFrame({"x": [1, np.nan], "y": [2, 3]})
    with pytest.raises(ValueError, match="2 paired"):
        pearson(df, "x", "y")


def test_pearson_no_variance() -> None:
    df = pd.DataFrame({"x": [1, 1, 1], "y": [1, 2, 3]})
    with pytest.raises(ValueError):
        pearson(df, "x", "y")


# --------------------------------------------------------------------------- #
# group_statistics — §2.5 t-CI vs z-CI
# --------------------------------------------------------------------------- #
def test_group_statistics_fields() -> None:
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1.0, 3.0, 10.0, 12.0]})
    groups = group_statistics(df, "cat", "val")
    assert len(groups) == 2
    by_cat = {g.category: g for g in groups}
    assert by_cat["a"].n == 2
    assert by_cat["a"].mean == pytest.approx(2.0)
    assert by_cat["b"].mean == pytest.approx(11.0)


def test_group_statistics_ci_uses_t_not_z_small_n() -> None:
    """§2.5: for small n the CI half-width must use t, not 1.96 (z)."""
    df = pd.DataFrame({"cat": ["a"] * 3, "val": [1.0, 2.0, 3.0]})
    groups = group_statistics(df, "cat", "val")
    g = groups[0]
    assert g.n == 3
    sem = g.sem
    expected_half = scipy_stats.t.ppf(0.975, df=2) * sem
    z_half = 1.96 * sem
    assert g.ci_upper - g.mean == pytest.approx(expected_half)
    # And explicitly NOT the z value (they differ for small n).
    assert not math.isclose(g.ci_upper - g.mean, z_half)


def test_group_statistics_single_observation_ci_collapses() -> None:
    df = pd.DataFrame({"cat": ["a"], "val": [5.0]})
    groups = group_statistics(df, "cat", "val")
    g = groups[0]
    assert g.n == 1
    assert g.sem == 0.0
    assert g.ci_lower == g.ci_upper == g.mean


def test_group_statistics_empty_returns_empty() -> None:
    df = pd.DataFrame({"cat": [], "val": []})
    assert group_statistics(df, "cat", "val") == ()


def test_group_statistics_skips_all_nan_group() -> None:
    # A group whose only value is NaN -> dropped by dropna; the guard skips it.
    df = pd.DataFrame({"cat": ["a", "a", "b"], "val": [np.nan, np.nan, 1.0]})
    groups = group_statistics(df, "cat", "val")
    assert len(groups) == 1
    assert groups[0].category == "b"


# --------------------------------------------------------------------------- #
# anova — §2.3 p-value by row name, §2.4 SettingWithCopyWarning
# --------------------------------------------------------------------------- #
def test_anova_significant_groups() -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "cat": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
            "val": np.concatenate(
                [rng.normal(0, 1, 30), rng.normal(5, 1, 30), rng.normal(10, 1, 30)]
            ),
        }
    )
    result = anova(df, "cat", "val")
    assert result.cat_var == "cat"
    assert result.num_var == "val"
    assert result.p_value < 0.05
    assert result.significant is True
    # The ANOVA table carries the C(cat) row by name.
    assert "C(cat)" in result.table.index


def test_anova_p_value_read_by_row_name_not_position() -> None:
    """§2.3: read via .loc['C(cat)'], never positionally.

    With typ=2 and a single factor the C(cat) row is not always row 0 (an
    intercept/residual ordering could shift it). We assert the result's
    p_value equals the row-NAME lookup, and that it's a valid float.
    """
    df = pd.DataFrame(
        {"cat": ["a", "a", "b", "b", "c", "c"], "val": [1.0, 2.0, 4.0, 5.0, 9.0, 10.0]}
    )
    result = anova(df, "cat", "val")
    expected = float(result.table.loc["C(cat)", "PR(>F)"])
    assert result.p_value == expected
    assert 0.0 < result.p_value < 0.05


def test_anova_insignificant_groups() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "cat": ["a"] * 30 + ["b"] * 30,
            "val": np.concatenate([rng.normal(0, 1, 30), rng.normal(0, 1, 30)]),
        }
    )
    result = anova(df, "cat", "val")
    assert result.p_value > 0.05
    assert result.significant is False


def test_anova_returns_group_stats_with_t_ci() -> None:
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1.0, 3.0, 10.0, 12.0]})
    result = anova(df, "cat", "val")
    assert len(result.group_stats) == 2
    for g in result.group_stats:
        half = g.ci_upper - g.mean
        expected = scipy_stats.t.ppf(0.975, df=g.n - 1) * g.sem
        assert half == pytest.approx(expected)


def test_anova_returns_tukey_rows() -> None:
    df = pd.DataFrame(
        {"cat": ["a", "a", "b", "b", "c", "c"], "val": [1.0, 2.0, 4.0, 5.0, 9.0, 10.0]}
    )
    result = anova(df, "cat", "val")
    assert len(result.tukey_rows) >= 1
    row = result.tukey_rows[0]
    # Tukey summary columns include group1, group2, meandiff, p-adj, ...
    assert "group1" in row
    assert "group2" in row
    assert "p-adj" in row


def test_anova_too_few_groups_raises() -> None:
    df = pd.DataFrame({"cat": ["a", "a"], "val": [1.0, 2.0]})
    with pytest.raises(ValueError, match="2 groups"):
        anova(df, "cat", "val")


def test_anova_no_observations_raises() -> None:
    df = pd.DataFrame({"cat": ["a", None], "val": [np.nan, 2.0]})
    with pytest.raises(ValueError, match="No complete"):
        anova(df, "cat", "val")


def test_anova_no_settingwithcopywarning() -> None:
    """§2.4: the .astype('category') on a slice must not warn."""
    df = pd.DataFrame({"cat": ["a", "a", "b", "b"], "val": [1.0, 2.0, 3.0, 4.0]})
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> failure
        # pandas may emit its own unrelated warnings; narrow to the target.
        warnings.filterwarnings(
            "error",
            message=".*SettingWithCopy.*",
        )
        anova(df, "cat", "val")
