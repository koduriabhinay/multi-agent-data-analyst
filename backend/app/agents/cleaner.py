"""
CleanerAgent — makes the data safe to analyse, and writes down what it changed.

Every transformation appends to `state["transformations"]`. That log is what
lets the final report say "we dropped 14 duplicate rows" instead of quietly
changing the numbers and hoping nobody checks.

No LLM call here. Cleaning rules should be deterministic and reproducible.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.agents.base import BaseAgent
from app.workflow.state import WorkflowState

log = logging.getLogger(__name__)

#: Above this fraction of nulls, imputing invents more data than it recovers.
DROP_COLUMN_NULL_THRESHOLD = 0.60

#: Standard Tukey fence multiplier for outlier flagging.
IQR_MULTIPLIER = 1.5


class CleanerAgent(BaseAgent):
    name = "cleaner"

    def run(self, state: WorkflowState) -> WorkflowState:
        df = state["raw_df"].copy()
        changes: list[dict[str, Any]] = []
        rows_before = len(df)

        df = self._drop_empty_columns(df, changes)
        df = self._drop_duplicates(df, changes)
        df = self._coerce_types(df, changes)
        df = self._impute_missing(df, changes)
        self._flag_outliers(df, changes)

        state["clean_df"] = df
        state["transformations"] = changes
        self.note = f"{rows_before} → {len(df)} rows, {len(changes)} transformations"
        return state

    # ------------------------------------------------------------------
    def _drop_empty_columns(self, df: pd.DataFrame, changes: list) -> pd.DataFrame:
        """Remove columns that are mostly or entirely missing."""
        if df.empty:
            return df

        null_frac = df.isnull().mean()
        to_drop = null_frac[null_frac > DROP_COLUMN_NULL_THRESHOLD].index.tolist()

        if to_drop:
            df = df.drop(columns=to_drop)
            changes.append(
                {
                    "step": "drop_sparse_columns",
                    "columns": to_drop,
                    "detail": f"Dropped columns more than {DROP_COLUMN_NULL_THRESHOLD:.0%} empty.",
                }
            )
        return df

    def _drop_duplicates(self, df: pd.DataFrame, changes: list) -> pd.DataFrame:
        n_dupes = int(df.duplicated().sum())
        if n_dupes:
            df = df.drop_duplicates().reset_index(drop=True)
            changes.append(
                {
                    "step": "drop_duplicates",
                    "rows_removed": n_dupes,
                    "detail": f"Removed {n_dupes} exact duplicate rows.",
                }
            )
        return df

    def _coerce_types(self, df: pd.DataFrame, changes: list) -> pd.DataFrame:
        """Turn date-like and number-like object columns into real dtypes."""
        converted: dict[str, str] = {}

        for col in df.select_dtypes(include=["object"]).columns:
            sample = df[col].dropna()
            if sample.empty:
                continue

            # Try dates first — "2024-01-15" would parse as neither int nor float
            if _looks_like_dates(sample):
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() > 0.8:
                    df[col] = parsed
                    converted[col] = "datetime64[ns]"
                    continue

            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().mean() > 0.8:
                df[col] = numeric
                converted[col] = "numeric"

        if converted:
            changes.append(
                {
                    "step": "convert_types",
                    "columns": converted,
                    "detail": f"Converted {len(converted)} text columns to typed values.",
                }
            )
        return df

    def _impute_missing(self, df: pd.DataFrame, changes: list) -> pd.DataFrame:
        """Fill remaining gaps: median for numbers, mode for categories."""
        filled: dict[str, str] = {}

        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            if n_missing == 0:
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                # Median, not mean — skewed columns shouldn't drag the fill value
                value = df[col].median()
                df[col] = df[col].fillna(value)
                filled[col] = f"median ({value:.4g}), {n_missing} values"
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].ffill().bfill()
                filled[col] = f"forward fill, {n_missing} values"
            else:
                modes = df[col].mode()
                value = modes.iloc[0] if not modes.empty else "Unknown"
                df[col] = df[col].fillna(value)
                filled[col] = f"mode ({value}), {n_missing} values"

        if filled:
            changes.append(
                {
                    "step": "impute_missing",
                    "columns": filled,
                    "detail": f"Filled missing values in {len(filled)} columns.",
                }
            )
        return df

    def _flag_outliers(self, df: pd.DataFrame, changes: list) -> None:
        """Count outliers per column but leave the rows in place.

        Extreme values are often the interesting part of a dataset. Flag them
        for the report; don't silently delete them.
        """
        flagged: dict[str, int] = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue

            low, high = q1 - IQR_MULTIPLIER * iqr, q3 + IQR_MULTIPLIER * iqr
            count = int(((series < low) | (series > high)).sum())
            if count:
                flagged[col] = count

        if flagged:
            changes.append(
                {
                    "step": "flag_outliers",
                    "columns": flagged,
                    "detail": "Outliers flagged using the IQR rule. Rows were kept.",
                }
            )


def _looks_like_dates(sample: pd.Series) -> bool:
    """Cheap check before paying for a full to_datetime parse."""
    text = sample.astype(str).head(20)
    hits = text.str.contains(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}", regex=True)
    return bool(hits.mean() > 0.5)
