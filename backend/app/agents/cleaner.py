"""
CleanerAgent — makes the data safe to analyse, and writes down what it changed.

Every transformation appends to `state["transformations"]`. That log is what
lets the final report say "we dropped 14 duplicate rows" instead of quietly
changing the numbers and hoping nobody checks.

No LLM call here. Cleaning rules should be deterministic and reproducible.
"""

from __future__ import annotations

import hashlib
import logging
import re
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

#: Name of the synthetic reference column added when a file has no natural
#: identifier. Reserved — Planner and Analyzer both know to exclude it from
#: statistics by this exact name, so don't rename it without updating them.
ROW_REF_COLUMN = "row_ref"

#: A column already looks like an identifier if its name suggests one...
ID_LIKE_NAME = re.compile(r"(?i)(^|_)(id|uuid|guid|key)($|_)|name$")

#: ...or if almost every value is unique, whatever it's called — but only
#: trusted for whole-number/text columns with enough rows to mean something.
#: A float column (salary, a credit score with decimals) looks "unique" at
#: almost any size purely because it's continuous data, not because it's an
#: identifier — and at a handful of rows, even an int column looks unique by
#: chance. Neither should suppress the reference column.
ID_LIKE_UNIQUENESS = 0.95
ID_LIKE_MIN_ROWS = 20

#: base36 length for the synthetic reference. 6 chars = 36^6 ≈ 2.18 billion
#: combinations — collisions between genuinely different rows are extended
#: automatically if they ever happen (see _ensure_row_reference), but at this
#: length they won't for any file this project's 10 MB cap will admit.
ROW_REF_LENGTH = 6
ROW_REF_MAX_LENGTH = 12


class CleanerAgent(BaseAgent):
    name = "cleaner"

    def run(self, state: WorkflowState) -> WorkflowState:
        df = state["raw_df"].copy()
        changes: list[dict[str, Any]] = []
        rows_before = len(df)

        df = self._drop_empty_columns(df, changes)
        df = self._drop_duplicates(df, changes)
        df = self._ensure_row_reference(df, changes)
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

    def _ensure_row_reference(self, df: pd.DataFrame, changes: list) -> pd.DataFrame:
        """Give every row a stable label, if nothing already does that job.

        Exact-duplicate detection compares every column, so it never needs an
        ID to work correctly — two different people can't become "the same
        row" just because an ID column exists to tell them apart. What an ID
        actually helps with is everything downstream: pointing at a specific
        row in a report ("row A3F91K was flagged as an outlier") when the
        file has nothing else to call it.

        Deliberately content-derived, not random: a random ID assigned before
        dedup would make every row unique by construction and silently
        disable duplicate detection. A hash of the row's own values doesn't
        have that problem — identical rows still hash identically, so this
        step changes nothing about what dedup catches. It also means the
        same file produces the same IDs on every run.
        """
        if df.empty or _has_identifier_column(df):
            return df

        # Row-wise, so cost scales with rows — fine at this project's 10 MB
        # upload cap, worth knowing if that cap ever changes.
        # Not df.astype(str) — on newer pandas that can leave NaN as an actual
        # float rather than stringifying it (dtype-dependent), which breaks
        # the join on any column with missing values. Python's own str() has
        # no such ambiguity.
        canonical = df.apply(lambda row: "\x1f".join(str(v) for v in row), axis=1)
        digests = canonical.map(lambda s: hashlib.sha256(s.encode()).hexdigest())

        length = ROW_REF_LENGTH
        while length <= ROW_REF_MAX_LENGTH:
            ids = digests.map(lambda h, n=length: _base36(int(h[:16], 16), n))
            if not ids.duplicated().any():
                break
            length += 1  # a truncation collision between two *different* rows
        else:
            log.warning(
                "Could not find collision-free row references up to %d chars; "
                "some rows may share a reference label.",
                ROW_REF_MAX_LENGTH,
            )

        df = df.copy()
        df.insert(0, ROW_REF_COLUMN, ids.values)

        changes.append(
            {
                "step": "add_row_reference",
                "column": ROW_REF_COLUMN,
                "detail": (
                    f"No identifier column found, so added '{ROW_REF_COLUMN}' "
                    f"({length}-character reference derived from each row's own "
                    f"values) so individual rows can be referred to in the report."
                ),
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


def _has_identifier_column(df: pd.DataFrame) -> bool:
    """Does this file already have something that tells rows apart?

    Checked by name (id, uuid, key, anything ending in "name") and by shape
    (almost every value unique, whatever the column is called) — either is
    enough, so an "account_number" column with no obvious name-match still
    counts if its values are in fact all distinct.
    """
    n = len(df)
    if n <= 1:
        return True  # nothing to disambiguate

    for col in df.columns:
        if ID_LIKE_NAME.search(str(col)):
            return True

        if (
            n >= ID_LIKE_MIN_ROWS
            and not pd.api.types.is_float_dtype(df[col])
            and df[col].nunique(dropna=True) / n >= ID_LIKE_UNIQUENESS
        ):
            return True

    return False


_BASE36_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _base36(n: int, length: int) -> str:
    """Encode a non-negative int as uppercase base36, padded/truncated to `length`.

    Base36 rather than raw hex so the label reads like a normal reference code
    (order confirmations, license plates) instead of looking like a hash.
    """
    if n == 0:
        digits = "0"
    else:
        digits = ""
        while n:
            n, rem = divmod(n, 36)
            digits = _BASE36_ALPHABET[rem] + digits

    return digits[-length:].rjust(length, "0")
