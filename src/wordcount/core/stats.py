"""Pure statistics: Pearson correlation, one-way ANOVA, group descriptives.

Replaces the *math* half of ``app/stats_ui.py`` (audit §2.3, §2.4, §2.5). The
Streamlit rendering half is gone; the functions return typed dataclasses from
:mod:`core.models` so the API/CLI/notebook all consume the same result.

Fixes:

* **§2.3 — ANOVA p-value by row name.** The legacy
  ``anova_table["PR(>F)"][0]`` was positional and would break if statsmodels
  ever reordered rows (intercept, covariates). We read
  ``anova_table.loc[f"C({cat_var})", "PR(>F)"]`` by name.
* **§2.4 — SettingWithCopyWarning.** The legacy
  ``df_clean = enhanced_df[[cat_var, num_var]].dropna()`` then
  ``df_clean[cat_var] = ... .astype("category")`` mutated a slice copy. We
  ``.copy()`` before ``.astype``.
* **§2.5 — t-CI not z-CI.** The legacy ``1.96 * sem`` used the normal
  approximation, wrong for small n. We use ``scipy.stats.t.ppf(0.975, df=n-1)``
  per group (the typical use case here).

Pure: no streamlit, no fastapi, no pydantic. ``pandas``/``scipy``/``statsmodels``
are the only deps (already core runtime deps).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from wordcount.core.models import AnovaResult, GroupStats, PearsonResult

#: Minimum paired observations / groups for the stats procedures.
_MIN_PAIRED_N: int = 2
_MIN_GROUPS: int = 2
#: Default ANOVA significance threshold.
_SIGNIFICANCE_ALPHA: float = 0.05
#: Default CI confidence level.
_DEFAULT_CONFIDENCE: float = 0.95


def pearson(df: pd.DataFrame, col1: str, col2: str) -> PearsonResult:
    """Pearson correlation between two numeric columns, listwise NA deletion.

    Raises :class:`ValueError` if the columns are equal, missing, or yield fewer
    than 2 paired non-NA observations (no variance). Returns a frozen
    :class:`PearsonResult` with ``coefficient``, ``p_value``, ``n``.
    """
    if col1 == col2:
        raise ValueError(f"Pearson requires two distinct columns; got {col1!r} twice.")
    pair = df[[col1, col2]].dropna()
    n = len(pair)
    if n < _MIN_PAIRED_N:
        raise ValueError(f"Pearson needs at least 2 paired non-NA observations; got {n}.")
    # Guard zero-variance columns: scipy.pearsonr raises a ConstantInput
    # warning (turned error under the project's filterwarnings=["error"]).
    if pair[col1].std(ddof=0) == 0.0 or pair[col2].std(ddof=0) == 0.0:
        raise ValueError(
            f"Pearson requires non-constant columns; {col1!r} or {col2!r} has no variance."
        )
    coefficient, p_value = stats.pearsonr(pair[col1], pair[col2])
    return PearsonResult(
        col1=col1,
        col2=col2,
        coefficient=float(coefficient),
        p_value=float(p_value),
        n=n,
    )


def _t_ci_half_width(sem: float, n: int, confidence: float = _DEFAULT_CONFIDENCE) -> float:
    """Half-width of a two-sided t-CI (§2.5: t, not 1.96)."""
    if n < _MIN_PAIRED_N or sem == 0.0:
        return 0.0
    alpha = 1.0 - confidence
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df=n - 1)
    return float(t_crit * sem)


def group_statistics(df: pd.DataFrame, cat_var: str, num_var: str) -> tuple[GroupStats, ...]:
    """Per-group ``n``, ``mean``, ``sem``, and a t-based 95% CI (fixes §2.5).

    Groups with fewer than 1 observation are skipped; SEM is 0 (and the CI
    collapses to the mean) when ``n == 1`` or the group has zero variance.
    """
    clean = df[[cat_var, num_var]].dropna()
    if clean.empty:
        return ()
    rows: list[GroupStats] = []
    for category, group in clean.groupby(cat_var, observed=True)[num_var]:
        n = int(group.count())
        mean = float(group.mean())
        sem = float(group.sem())
        if math.isnan(sem):  # n == 1 -> ddof=1 SEM is NaN; treat as 0.
            sem = 0.0
        half = _t_ci_half_width(sem, n)
        rows.append(
            GroupStats(
                category=str(category),
                n=n,
                mean=mean,
                sem=sem,
                ci_lower=mean - half,
                ci_upper=mean + half,
            )
        )
    return tuple(rows)


def anova(df: pd.DataFrame, cat_var: str, num_var: str) -> AnovaResult:
    """One-way ANOVA of ``num_var`` across the groups of ``cat_var``.

    Reads the p-value by **row name** (``C({cat_var})``) — fixes §2.3. Copies
    the cleaned frame before ``.astype("category")`` — fixes §2.4. Returns the
    full statsmodels ANOVA table, a per-group descriptive tuple (t-CI, §2.5),
    Tukey HSD rows, and ``significant`` mirroring ``p_value < 0.05``.

    Raises :class:`ValueError` if there are fewer than 2 groups or no complete
    observations.
    """
    clean = df[[cat_var, num_var]].dropna().copy()  # §2.4: .copy() before .astype
    if clean.empty:
        raise ValueError("No complete observations for ANOVA.")
    clean[cat_var] = clean[cat_var].astype("category")

    n_groups = clean[cat_var].nunique()
    if n_groups < _MIN_GROUPS:
        raise ValueError(f"ANOVA needs at least 2 groups in {cat_var!r}; got {n_groups}.")

    model = ols(f"{num_var} ~ C({cat_var})", data=clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    row_name = f"C({cat_var})"
    p_value = float(anova_table.loc[row_name, "PR(>F)"])

    tukey = pairwise_tukeyhsd(endog=clean[num_var], groups=clean[cat_var], alpha=0.05)
    summary = tukey.summary()
    tukey_rows: tuple[Mapping[str, Any], ...] = tuple(
        dict(zip(summary.data[0], row, strict=True)) for row in summary.data[1:]
    )

    group_stats = group_statistics(clean, cat_var, num_var)

    return AnovaResult(
        cat_var=cat_var,
        num_var=num_var,
        p_value=p_value,
        significant=bool(p_value < _SIGNIFICANCE_ALPHA),
        table=anova_table,
        group_stats=group_stats,
        tukey_rows=tukey_rows,
    )


__all__ = ["anova", "group_statistics", "pearson"]
