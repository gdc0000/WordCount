"""Pure Plotly figure builders (no rendering, no streamlit).

These return ``plotly.graph_objects.Figure`` objects. The API serializes them
via ``fig.to_dict()`` so any UX renders Plotly client-side without bundling a
plotly server. ``viz/`` is pure: no ``st.plotly_chart``, no ``st.warning``.

Three builders:

* :func:`bar_figure` — horizontal bar of top-N detected terms for a category,
  fed by :func:`viz.plots.build_barplot_data`.
* :func:`scatter_with_trend` — scatter of two numeric columns with an OLS
  trendline (replaces the legacy inline ``px.scatter(..., trendline="ols")``).
* :func:`anova_bar_figure` — horizontal bar of group means with t-based 95% CI
  error bars (§2.5: the CI is computed in :mod:`core.stats`, the figure just
  renders it).
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wordcount.core.models import GroupStats


def bar_figure(
    bar_data: pd.DataFrame,
    *,
    label: str,
    top_n: int,
) -> go.Figure:
    """Horizontal bar of top-N detected terms for a category.

    ``bar_data`` is the long-form frame from :func:`viz.plots.build_barplot_data`
    (columns ``term``, ``frequency``, ``doc_frequency``).
    """
    fig = px.bar(
        bar_data,
        x="frequency",
        y="term",
        orientation="h",
        title=f"Top {top_n} terms in '{label}'",
        labels={"frequency": "Frequency (occurrences)", "term": "Term"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def scatter_with_trend(
    df: pd.DataFrame,
    *,
    col1: str,
    col2: str,
) -> go.Figure:
    """Scatter of two numeric columns with an OLS trendline.

    ``df`` must already be listwise-dropped on ``[col1, col2]`` (callers handle
    NA deletion via :mod:`core.stats.pearson` semantics).
    """
    fig = px.scatter(
        df,
        x=col1,
        y=col2,
        trendline="ols",
        title=f"{col2} vs {col1}",
    )
    return fig


def anova_bar_figure(
    group_stats: tuple[GroupStats, ...],
    *,
    cat_var: str,
    num_var: str,
) -> go.Figure:
    """Horizontal bar of group means with t-based 95% CI error bars (§2.5).

    Renders the CIs computed in :func:`core.stats.group_statistics` — the
    figure does no math, only rendering.
    """
    if not group_stats:
        fig = go.Figure()
        fig.update_layout(title=f"No groups for ANOVA of {num_var} by {cat_var}")
        return fig

    categories = [g.category for g in group_stats]
    means = [g.mean for g in group_stats]
    ci_lower = [g.ci_lower for g in group_stats]
    ci_upper = [g.ci_upper for g in group_stats]
    # plotly error_x is the half-width symmetric around the point.
    err_plus = [upper - mean for upper, mean in zip(ci_upper, means, strict=True)]
    err_minus = [mean - lower for mean, lower in zip(means, ci_lower, strict=True)]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=means,
            y=categories,
            orientation="h",
            error_x=dict(type="data", array=err_plus, arrayminus=err_minus),
            name=f"Mean {num_var}",
        )
    )
    fig.update_layout(
        title=f"Mean {num_var} by {cat_var} with 95% CI",
        xaxis_title=f"Mean of {num_var}",
        yaxis_title=cat_var,
        yaxis={"categoryorder": "total ascending"},
    )
    return fig


__all__ = ["anova_bar_figure", "bar_figure", "scatter_with_trend"]
