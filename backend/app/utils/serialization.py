"""
Making pandas and numpy output safe to serialise.

`json.dumps` rejects NaN, infinity, numpy scalars, and Timestamps — all of
which come out of a dataframe by default. Sanitising once on the way into
the database means every downstream consumer gets clean JSON.
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd


def json_safe(value: Any) -> Any:
    """Recursively convert a value into something `json.dumps` accepts.

    NaN and infinity become None, since JSON has no representation for them
    and null is what a consumer would do with them anyway.
    """
    # Check for null first — pd.isna handles NaN, NaT, and None in one call
    if value is None:
        return None

    if isinstance(value, float | np.floating):
        number = float(value)
        return None if (math.isnan(number) or math.isinf(number)) else number

    if isinstance(value, bool | np.bool_):
        return bool(value)

    if isinstance(value, int | np.integer):
        return int(value)

    if isinstance(value, pd.Timestamp | datetime | date):
        return None if pd.isna(value) else value.isoformat()

    if isinstance(value, np.datetime64):
        converted = pd.Timestamp(value)
        return None if pd.isna(converted) else converted.isoformat()

    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]

    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list | tuple | set):
        return [json_safe(v) for v in value]

    if isinstance(value, str):
        return value

    # pd.isna raises on some container types, so this stays last
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    return value
