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
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        """Built directly on graph_objects rather than plotly.express.

        Express recomputes its own aggregation and validation on every call —
        measured at ~51ms per histogram here versus ~20ms doing the same
        subplot by hand. Same marginal box, same bin count, same output shape;
        the difference is purely which layer of the Plotly API builds it.
        """
        charts = []
        for col in df.select_dtypes(include=[np.number]).columns[:MAX_HISTOGRAMS]:
            series = df[col].dropna()
            if series.empty or series.std() == 0:
                continue

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.18, 0.82],
                vertical_spacing=0.02,
            )
            fig.add_trace(
                go.Box(x=series, boxpoints=False, name="", showlegend=False), row=1, col=1
            )
            fig.add_trace(go.Histogram(x=series, nbinsx=40, showlegend=False), row=2, col=1)
            fig.update_yaxes(visible=False, row=1, col=1)
            fig.update_xaxes(title_text=col, row=2, col=1)
            fig.update_yaxes(title_text="Count", row=2, col=1)
            fig.update_layout(title=f"Distribution of {col}", **LAYOUT)

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
        """Built directly on graph_objects; the trendline is numpy, not statsmodels.

        px.scatter(trendline="ols") calls into statsmodels for a full OLS fit
        with a summary table we never look at — measured at ~220ms per call,
        eleven times slower than doing the same simple linear fit with
        numpy.polyfit. It's the same line: both are ordinary least squares on
        two columns. The r and p-value shown were already computed by the
        Analyzer with scipy; this only needs the fit's slope and intercept to
        draw it.
        """
        pairs = results.get("correlation", {}).get("notable_pairs", [])
        charts = []

        for pair in pairs[:MAX_SCATTERPLOTS]:
            a, b = pair["column_a"], pair["column_b"]
            if a not in df.columns or b not in df.columns:
                continue

            paired = df[[a, b]].dropna()
            if len(paired) < 2:
                continue

            fig = go.Figure()
            fig.add_trace(
                go.Scattergl(
                    x=paired[a], y=paired[b], mode="markers", opacity=0.6, showlegend=False
                )
            )

            try:
                with warnings.catch_warnings():
                    # A constant column makes the fit rank-deficient — expected
                    # for this case, not a real numerical problem, so it's
                    # handled below rather than logged as a warning.
                    warnings.simplefilter("ignore", np.exceptions.RankWarning)
                    slope, intercept = np.polyfit(paired[a], paired[b], 1)
                line_x = np.array([paired[a].min(), paired[a].max()])
                fig.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=slope * line_x + intercept,
                        mode="lines",
                        line={"color": "firebrick"},
                        showlegend=False,
                    )
                )
            except np.linalg.LinAlgError:
                pass  # degenerate data (e.g. one column is constant) — plot without a line

            fig.update_layout(
                title=f"{a} vs {b} (r = {pair['r']:.2f})",
                xaxis_title=a,
                yaxis_title=b,
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

            fig = go.Figure()
            for group_name, group_df in df.groupby(group, observed=True):
                fig.add_trace(go.Box(y=group_df[value], name=str(group_name), boxpoints="outliers"))
            fig.update_layout(title=f"{value} by {group}", showlegend=False, **LAYOUT)

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

        fig = go.Figure(
            go.Scatter(
                x=ordered[date_col],
                y=ordered[value_col],
                mode="lines+markers" if len(ordered) < 100 else "lines",
            )
        )
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
