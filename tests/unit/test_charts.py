"""Tests for ``wordcount.viz.charts`` — pure figure builders (no rendering).

Asserts the Figure structure (traces, layout, orientation) without rendering.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import pytest

from wordcount.core.models import GroupStats
from wordcount.viz.charts import anova_bar_figure, bar_figure, scatter_with_trend


# --------------------------------------------------------------------------- #
# bar_figure
# --------------------------------------------------------------------------- #
def test_bar_figure_returns_figure() -> None:
    bar_data = pd.DataFrame(
        {"term": ["happy", "sad"], "frequency": [3, 1], "doc_frequency": [2, 1]}
    )
    fig = bar_figure(bar_data, label="Affect", top_n=2)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    # Horizontal bar => orientation "h".
    assert fig.data[0].orientation == "h"


def test_bar_figure_title_includes_label_and_top_n() -> None:
    bar_data = pd.DataFrame({"term": ["x"], "frequency": [1], "doc_frequency": [1]})
    fig = bar_figure(bar_data, label="Place", top_n=5)
    assert "Place" in fig.layout.title.text
    assert "5" in fig.layout.title.text


def test_bar_figure_yaxis_ordered_total_ascending() -> None:
    bar_data = pd.DataFrame({"term": ["a", "b"], "frequency": [1, 2], "doc_frequency": [1, 1]})
    fig = bar_figure(bar_data, label="Cat", top_n=2)
    assert fig.layout.yaxis.categoryorder == "total ascending"


# --------------------------------------------------------------------------- #
# scatter_with_trend
# --------------------------------------------------------------------------- #
def test_scatter_with_trend_returns_figure_with_trendline() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    fig = scatter_with_trend(df, col1="x", col2="y")
    assert isinstance(fig, go.Figure)
    # px.scatter with trendline="ols" adds a second trace (the trendline).
    assert len(fig.data) >= 2


def test_scatter_with_trend_title() -> None:
    df = pd.DataFrame({"x": [1, 2], "y": [1, 2]})
    fig = scatter_with_trend(df, col1="x", col2="y")
    assert "y" in fig.layout.title.text
    assert "x" in fig.layout.title.text


# --------------------------------------------------------------------------- #
# anova_bar_figure — §2.5 renders the t-CI from group_stats
# --------------------------------------------------------------------------- #
def test_anova_bar_figure_renders_group_means_with_ci() -> None:
    group_stats = (
        GroupStats(category="a", n=3, mean=2.0, sem=0.5, ci_lower=1.0, ci_upper=3.0),
        GroupStats(category="b", n=3, mean=10.0, sem=0.5, ci_lower=9.0, ci_upper=11.0),
    )
    fig = anova_bar_figure(group_stats, cat_var="cat", num_var="val")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.orientation == "h"
    # Means rendered as x, categories as y.
    assert list(trace.x) == [2.0, 10.0]
    assert list(trace.y) == ["a", "b"]
    # Error bars present (half-widths).
    assert trace.error_x is not None
    assert trace.error_x.array is not None
    assert trace.error_x.arrayminus is not None


def test_anova_bar_figure_ci_half_widths_match_stats() -> None:
    group_stats = (GroupStats(category="a", n=3, mean=2.0, sem=0.5, ci_lower=1.0, ci_upper=3.0),)
    fig = anova_bar_figure(group_stats, cat_var="cat", num_var="val")
    trace = fig.data[0]
    # ci_upper - mean = 1.0; mean - ci_lower = 1.0
    assert trace.error_x.array[0] == pytest.approx(1.0)
    assert trace.error_x.arrayminus[0] == pytest.approx(1.0)


def test_anova_bar_figure_empty_groups() -> None:
    fig = anova_bar_figure((), cat_var="cat", num_var="val")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_anova_bar_figure_title_and_axes() -> None:
    group_stats = (GroupStats(category="a", n=2, mean=1.0, sem=0.1, ci_lower=0.5, ci_upper=1.5),)
    fig = anova_bar_figure(group_stats, cat_var="Group", num_var="Score")
    assert "Score" in fig.layout.title.text
    assert "Group" in fig.layout.title.text
    assert fig.layout.yaxis.categoryorder == "total ascending"
