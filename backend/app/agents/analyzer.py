"""
AnalyzerAgent — runs the statistics the Planner asked for.

Everything here is real scipy / sklearn work. Each analysis is a separate
method that returns a plain dict, so a failure in one (say, regression on a
column with no variance) doesn't lose the results of the others.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from app.agents.base import BaseAgent
from app.workflow.state import WorkflowState

log = logging.getLogger(__name__)

#: Shapiro-Wilk is unreliable above this sample size; subsample instead.
SHAPIRO_MAX_N = 5000

#: |r| at or above this counts as a notable relationship worth reporting.
STRONG_CORRELATION = 0.5


class AnalyzerAgent(BaseAgent):
    name = "analyzer"

    def run(self, state: WorkflowState) -> WorkflowState:
        df = state["clean_df"]
        plan = state.get("plan", {})
        requested = plan.get("analyses", [])
        results: dict[str, Any] = {}

        runners = {
            "descriptive_stats": self._descriptive,
            "correlation": self._correlation,
            "distribution_tests": self._distribution_tests,
            "group_comparison": self._group_comparison,
            "outlier_detection": self._anomalies,
        }

        for key, runner in runners.items():
            if key not in requested:
                continue
            try:
                output = runner(df)
                if output:
                    results[key] = output
            except Exception as exc:
                log.warning("Analysis '%s' failed: %s", key, exc)
                results[key] = {"error": str(exc)}

        if "regression" in requested and plan.get("target_column"):
            try:
                results["regression"] = self._regression(df, plan["target_column"])
            except Exception as exc:
                log.warning("Regression failed: %s", exc)

        state["results"] = results
        self.note = f"completed {len(results)} analyses"
        return state

    # ------------------------------------------------------------------
    def _descriptive(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number])
        summary: dict[str, Any] = {"numeric": {}, "categorical": {}}

        for col in numeric.columns:
            s = numeric[col].dropna()
            if s.empty:
                continue
            summary["numeric"][col] = {
                "count": int(s.count()),
                "mean": _f(s.mean()),
                "std": _f(s.std()),
                "min": _f(s.min()),
                "q25": _f(s.quantile(0.25)),
                "median": _f(s.median()),
                "q75": _f(s.quantile(0.75)),
                "max": _f(s.max()),
                "skew": _f(s.skew()),
                "kurtosis": _f(s.kurtosis()),
            }

        for col in df.select_dtypes(include=["object", "category"]).columns:
            s = df[col].dropna()
            if s.empty:
                continue
            top = s.value_counts().head(5)
            summary["categorical"][col] = {
                "unique": int(s.nunique()),
                "top_values": {str(k): int(v) for k, v in top.items()},
            }

        return summary

    def _correlation(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number])
        # Constant columns produce NaN correlations and clutter the heatmap
        numeric = numeric.loc[:, numeric.std() > 0]

        if numeric.shape[1] < 2:
            return {}

        matrix = numeric.corr(method="pearson")
        notable = []

        cols = matrix.columns.tolist()
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                r = matrix.loc[a, b]
                if pd.isna(r) or abs(r) < STRONG_CORRELATION:
                    continue

                # A correlation without a p-value is a shape, not a finding
                paired = df[[a, b]].dropna()
                _, p_value = stats.pearsonr(paired[a], paired[b])

                notable.append(
                    {
                        "column_a": a,
                        "column_b": b,
                        "r": _f(r),
                        "p_value": _f(p_value),
                        "significant": bool(p_value < 0.05),
                        "direction": "positive" if r > 0 else "negative",
                    }
                )

        notable.sort(key=lambda d: abs(d["r"]), reverse=True)

        return {
            "matrix": {c: {k: _f(v) for k, v in matrix[c].items()} for c in matrix.columns},
            "columns": cols,
            "notable_pairs": notable[:10],
        }

    def _distribution_tests(self, df: pd.DataFrame) -> dict[str, Any]:
        """Test each numeric column for normality — this drives later test choices."""
        tests = []

        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) < 8 or s.std() == 0:
                continue

            sample = s.sample(SHAPIRO_MAX_N, random_state=42) if len(s) > SHAPIRO_MAX_N else s
            statistic, p_value = stats.shapiro(sample)

            tests.append(
                {
                    "column": col,
                    "test": "Shapiro-Wilk",
                    "statistic": _f(statistic),
                    "p_value": _f(p_value),
                    "normally_distributed": bool(p_value > 0.05),
                    "n_tested": len(sample),
                }
            )

        return {"normality": tests}

    def _group_comparison(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compare numeric columns across categorical groups (t-test or ANOVA)."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        group_cols = [
            c
            for c in df.select_dtypes(include=["object", "category"]).columns
            if 1 < df[c].nunique() <= 20
        ]

        comparisons = []
        for group_col in group_cols[:3]:
            for value_col in numeric_cols[:5]:
                groups = [
                    g[value_col].dropna().values
                    for _, g in df.groupby(group_col, observed=True)
                    if len(g[value_col].dropna()) >= 3
                ]
                if len(groups) < 2:
                    continue

                if len(groups) == 2:
                    test_name = "Independent t-test"
                    statistic, p_value = stats.ttest_ind(*groups, equal_var=False)
                else:
                    test_name = "One-way ANOVA"
                    statistic, p_value = stats.f_oneway(*groups)

                comparisons.append(
                    {
                        "group_column": group_col,
                        "value_column": value_col,
                        "test": test_name,
                        "n_groups": len(groups),
                        "statistic": _f(statistic),
                        "p_value": _f(p_value),
                        "significant": bool(p_value < 0.05),
                    }
                )

        comparisons.sort(key=lambda d: d["p_value"])
        return {"comparisons": comparisons[:10]}

    def _anomalies(self, df: pd.DataFrame) -> dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number]).dropna()
        if numeric.shape[0] < 20 or numeric.shape[1] < 1:
            return {}

        model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
        labels = model.fit_predict(numeric)
        scores = model.score_samples(numeric)

        anomaly_positions = np.where(labels == -1)[0]

        return {
            "method": "Isolation Forest",
            "count": len(anomaly_positions),
            "percentage": _f(len(anomaly_positions) / len(numeric) * 100),
            "row_indices": [int(numeric.index[i]) for i in anomaly_positions[:50]],
            "mean_anomaly_score": _f(float(scores[anomaly_positions].mean()))
            if len(anomaly_positions)
            else None,
        }

    def _regression(self, df: pd.DataFrame, target: str) -> dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number]).dropna()
        if target not in numeric.columns or numeric.shape[1] < 2 or len(numeric) < 30:
            return {"skipped": "Not enough numeric data to fit a model."}

        features = numeric.drop(columns=[target])
        target_values = numeric[target]

        train_x, test_x, train_y, test_y = train_test_split(
            features, target_values, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)

        importance = sorted(
            (
                {"feature": f, "importance": _f(v)}
                for f, v in zip(features.columns, model.feature_importances_, strict=True)
            ),
            key=lambda d: d["importance"],
            reverse=True,
        )

        return {
            "target": target,
            "model": "Random Forest Regressor",
            "n_train": len(train_x),
            "n_test": len(test_x),
            "r2_score": _f(r2_score(test_y, predictions)),
            "mae": _f(mean_absolute_error(test_y, predictions)),
            "feature_importance": importance[:10],
        }


def _f(value: Any) -> float | None:
    """Convert numpy scalars to JSON-safe floats, mapping NaN/inf to None."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(result) or np.isinf(result)) else round(result, 6)
