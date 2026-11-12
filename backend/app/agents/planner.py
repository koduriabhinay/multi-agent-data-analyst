"""
PlannerAgent — looks at the shape of the data and decides what's worth running.

The profile is computed in pandas (cheap, deterministic). Only the *decision*
goes to the LLM, which keeps token usage low and means a bad model response
degrades to a sensible default rather than breaking the run.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.agents.base import BaseAgent
from app.workflow.state import WorkflowState

log = logging.getLogger(__name__)

VALID_ANALYSES = {
    "descriptive_stats",
    "correlation",
    "distribution_tests",
    "group_comparison",
    "outlier_detection",
    "regression",
}


def profile_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Compact summary of a dataframe — small enough to fit in a prompt."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime = df.select_dtypes(include=["datetime64[ns]", "datetime64"]).columns.tolist()

    null_counts = df.isnull().sum()

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_columns": datetime,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_counts": {c: int(n) for c, n in null_counts.items() if n > 0},
        "null_pct": {
            c: round(float(n) / len(df) * 100, 2)
            for c, n in null_counts.items()
            if n > 0 and len(df) > 0
        },
        # Low-cardinality categoricals are the ones worth grouping by
        "groupable_columns": [c for c in categorical if 1 < df[c].nunique() <= 20],
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


def heuristic_plan(profile: dict[str, Any]) -> dict[str, Any]:
    """What to run if the LLM is unavailable or returns nonsense."""
    analyses = ["descriptive_stats"]
    numeric = profile["numeric_columns"]

    if len(numeric) >= 2:
        analyses.append("correlation")
    if numeric:
        analyses.extend(["distribution_tests", "outlier_detection"])
    if profile["groupable_columns"] and numeric:
        analyses.append("group_comparison")

    return {
        "analyses": analyses,
        "target_column": None,
        "concerns": [],
        "notes": "Plan derived from dataset shape.",
    }


class PlannerAgent(BaseAgent):
    name = "planner"

    def run(self, state: WorkflowState) -> WorkflowState:
        df = state["raw_df"]
        profile = profile_dataframe(df)
        state["profile"] = profile

        fallback = heuristic_plan(profile)

        # Without a model, the heuristic plan is better than a canned response —
        # it at least reflects this dataset's actual shape.
        if self.llm.offline:
            state["plan"] = fallback
            self.note = f"planned {len(fallback['analyses'])} analyses (offline heuristics)"
            return state

        prompt = f"""You are planning a data analysis. Here is the dataset profile:

Rows: {profile["rows"]}
Columns: {profile["columns"]}
Numeric columns: {profile["numeric_columns"]}
Categorical columns: {profile["categorical_columns"]}
Datetime columns: {profile["datetime_columns"]}
Columns with missing values (% missing): {profile["null_pct"]}
Duplicate rows: {profile["duplicate_rows"]}
Low-cardinality columns suitable for grouping: {profile["groupable_columns"]}

Sample rows:
{profile["sample_rows"]}

Choose which analyses to run. Pick only from this list:
{sorted(VALID_ANALYSES)}

Return JSON with exactly these keys:
- "analyses": list of analysis names from the list above
- "target_column": a numeric column name to predict, or null if none is obvious
- "concerns": list of short strings describing data quality problems you noticed
- "notes": one sentence explaining your choices
"""

        plan = self.llm.ask_json(prompt, fallback=fallback)

        # Never trust the model's list blindly — drop anything unrecognised
        requested = plan.get("analyses") or []
        plan["analyses"] = [a for a in requested if a in VALID_ANALYSES]
        if not plan["analyses"]:
            log.warning("Planner returned no valid analyses; using heuristics")
            plan = fallback

        # A target column that isn't numeric can't be regressed on
        target = plan.get("target_column")
        if target and target not in profile["numeric_columns"]:
            plan["target_column"] = None

        state["plan"] = plan
        self.note = f"planned {len(plan['analyses'])} analyses"
        return state
