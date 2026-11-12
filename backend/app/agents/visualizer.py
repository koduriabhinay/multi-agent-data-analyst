"""
VisualizerAgent — turns analysis results into Plotly figures.

Charts are stored as Plotly JSON, not images. The frontend renders them
interactively, and the same payload works for a PNG export later.

Chart choice is driven by what the Analyzer actually found: a correlation
heatmap only appears if correlations were computed, scatter plots only for
pairs that cleared the significance threshold.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.agents.base import BaseAgent
from app.workflow.state import WorkflowState

log = logging.getLogger(__name__)

MAX_HISTOGRAMS = 6
MAX_SCATTERPLOTS = 3
LAYOUT = {"template": "plotly_white", "margin": {"l": 50, "r": 30, "t": 60, "b": 50}}


class VisualizerAgent(BaseAgent):
    name = "visualizer"

    def run(self, state: WorkflowState) -> WorkflowState:
        df = state["clean_df"]
        results = state.get("results", {})
        charts: list[dict] = []

        builders = [
            ("distributions", lambda: self._histograms(df)),
            ("correlation heatmap", lambda: self._heatmap(results)),
            ("scatter plots", lambda: self._scatters(df, results)),
            ("group boxplots", lambda: self._boxplots(df, results)),
            ("time series", lambda: self._timeseries(df)),
            ("feature importance", lambda: self._importance(results)),
        ]

        for label, builder in builders:
            try:
                charts.extend(builder())
            except Exception as exc:
                log.warning("Chart group '%s' failed: %s", label, exc)

        state["charts"] = charts
        self.note = f"built {len(charts)} charts"
        return state

    # ------------------------------------------------------------------
    def _histograms(self, df: pd.DataFrame) -> list[dict]:
        charts = []
        for col in df.select_dtypes(include=[np.number]).columns[:MAX_HISTOGRAMS]:
            series = df[col].dropna()
            if series.empty or series.std() == 0:
                continue

            fig = px.histogram(df, x=col, nbins=40, marginal="box")
            fig.update_layout(
                title=f"Distribution of {col}",
                xaxis_title=col,
                yaxis_title="Count",
                **LAYOUT,
            )
            charts.append(
                _chart(
                    "histogram",
                    f"Distribution of {col}",
                    f"Values range from {series.min():.4g} to {series.max():.4g}, "
                    f"centred on a median of {series.median():.4g}.",
                    fig,
                )
            )
        return charts

    def _heatmap(self, results: dict) -> list[dict]:
        correlation = results.get("correlation")
        if not correlation or not correlation.get("columns"):
            return []

        cols = correlation["columns"]
        matrix = [[correlation["matrix"][a].get(b) for b in cols] for a in cols]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=cols,
                y=cols,
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=[[f"{v:.2f}" if v is not None else "" for v in row] for row in matrix],
                texttemplate="%{text}",
                colorbar={"title": "r"},
            )
        )
        fig.update_layout(title="Correlation between numeric columns", **LAYOUT)

        return [
            _chart(
                "heatmap",
                "Correlation matrix",
                "Pearson correlation across every numeric column. "
                "Blue is negative, red is positive.",
                fig,
            )
        ]

    def _scatters(self, df: pd.DataFrame, results: dict) -> list[dict]:
        pairs = results.get("correlation", {}).get("notable_pairs", [])
        charts = []

        for pair in pairs[:MAX_SCATTERPLOTS]:
            a, b = pair["column_a"], pair["column_b"]
            if a not in df.columns or b not in df.columns:
                continue

            # The OLS trendline needs statsmodels; the chart is still useful without it
            try:
                fig = px.scatter(df, x=a, y=b, opacity=0.6, trendline="ols")
            except (ImportError, ModuleNotFoundError):
                fig = px.scatter(df, x=a, y=b, opacity=0.6)

            fig.update_layout(
                title=f"{a} vs {b} (r = {pair['r']:.2f})",
                **LAYOUT,
            )

            significance = "statistically significant" if pair["significant"] else "not significant"
            charts.append(
                _chart(
                    "scatter",
                    f"{a} vs {b}",
                    f"A {pair['direction']} relationship (r = {pair['r']:.2f}), {significance} "
                    f"at p = {pair['p_value']:.4g}.",
                    fig,
                )
            )
        return charts

    def _boxplots(self, df: pd.DataFrame, results: dict) -> list[dict]:
        comparisons = results.get("group_comparison", {}).get("comparisons", [])
        charts = []
        seen: set[tuple[str, str]] = set()

        for comparison in comparisons[:3]:
            group, value = comparison["group_column"], comparison["value_column"]
            if (group, value) in seen or group not in df.columns or value not in df.columns:
                continue
            seen.add((group, value))

            fig = px.box(df, x=group, y=value, points="outliers")
            fig.update_layout(title=f"{value} by {group}", **LAYOUT)

            verdict = (
                "The groups differ significantly."
                if comparison["significant"]
                else "No significant difference between groups."
            )
            charts.append(
                _chart(
                    "box",
                    f"{value} by {group}",
                    f"{comparison['test']} across {comparison['n_groups']} groups, "
                    f"p = {comparison['p_value']:.4g}. {verdict}",
                    fig,
                )
            )
        return charts

    def _timeseries(self, df: pd.DataFrame) -> list[dict]:
        date_cols = df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if len(date_cols) == 0 or len(numeric_cols) == 0:
            return []

        date_col, value_col = date_cols[0], numeric_cols[0]
        ordered = df[[date_col, value_col]].dropna().sort_values(date_col)
        if len(ordered) < 3:
            return []

        fig = px.line(ordered, x=date_col, y=value_col, markers=len(ordered) < 100)
        fig.update_layout(title=f"{value_col} over time", **LAYOUT)

        return [
            _chart(
                "line",
                f"{value_col} over time",
                f"Tracked from {ordered[date_col].min():%Y-%m-%d} to "
                f"{ordered[date_col].max():%Y-%m-%d}.",
                fig,
            )
        ]

    def _importance(self, results: dict) -> list[dict]:
        regression = results.get("regression", {})
        importance = regression.get("feature_importance")
        if not importance:
            return []

        # Horizontal bars read better for long feature names
        ordered = list(reversed(importance))
        fig = go.Figure(
            go.Bar(
                x=[f["importance"] for f in ordered],
                y=[f["feature"] for f in ordered],
                orientation="h",
            )
        )
        fig.update_layout(
            title=f"What predicts {regression['target']}",
            xaxis_title="Relative importance",
            **LAYOUT,
        )

        return [
            _chart(
                "bar",
                f"Feature importance for {regression['target']}",
                f"Random forest model, R² = {regression.get('r2_score')}.",
                fig,
            )
        ]


def _chart(kind: str, title: str, description: str, fig: go.Figure) -> dict:
    """Package a figure with the metadata the frontend and report agent need."""
    return {
        "chart_type": kind,
        "title": title,
        "description": description,
        "plotly_json": json.loads(fig.to_json()),
    }
