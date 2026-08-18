#!/usr/bin/env python3
"""
Generate sample datasets with realistic problems baked in.

The data has genuine signal (correlations, group differences) plus the mess
you'd get from a real export: missing values, duplicate rows, numbers stored
as text, outliers. That way the pipeline has something to actually find.

    python scripts/make_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[2] / "data_samples"
RNG = np.random.default_rng(42)


def employees(n: int = 600) -> pd.DataFrame:
    """Salary data where tenure and level genuinely predict pay."""
    departments = RNG.choice(
        ["Engineering", "Sales", "Marketing", "Support", "Finance"],
        size=n,
        p=[0.35, 0.25, 0.15, 0.15, 0.10],
    )
    levels = RNG.choice([1, 2, 3, 4, 5], size=n, p=[0.25, 0.30, 0.25, 0.15, 0.05])
    tenure = np.clip(RNG.gamma(2.0, 2.2, n), 0.1, 25).round(1)

    dept_premium = {
        "Engineering": 22_000,
        "Sales": 8_000,
        "Marketing": 4_000,
        "Support": 0,
        "Finance": 12_000,
    }

    salary = (
        52_000
        + levels * 17_500
        + tenure * 2_100
        + np.array([dept_premium[d] for d in departments])
        + RNG.normal(0, 7_500, n)
    ).round(-2)

    # Performance tracks level, loosely
    performance = np.clip(2.6 + levels * 0.28 + RNG.normal(0, 0.55, n), 1.0, 5.0).round(2)

    satisfaction = np.clip(
        6.9 + performance * 0.42 - tenure * 0.07 + RNG.normal(0, 1.1, n), 1.0, 10.0
    ).round(1)

    df = pd.DataFrame(
        {
            "employee_id": [f"E{i:05d}" for i in range(1, n + 1)],
            "department": departments,
            "level": levels,
            "tenure_years": tenure,
            "salary": salary,
            "performance_score": performance,
            "satisfaction": satisfaction,
            "hire_date": pd.to_datetime("2024-06-01")
            - pd.to_timedelta((tenure * 365).astype(int), unit="D"),
            "remote": RNG.choice(["Yes", "No"], n, p=[0.42, 0.58]),
        }
    )

    return _add_realistic_mess(df)


def sales(n: int = 900) -> pd.DataFrame:
    """Transactions with seasonality and a marketing-spend relationship."""
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(RNG.integers(0, 365, n), unit="D")
    month = dates.month

    # Q4 lift, February slump
    seasonal = np.where(month >= 10, 1.35, np.where(month == 2, 0.78, 1.0))

    marketing_spend = np.clip(RNG.gamma(3, 400, n), 50, 8000).round(2)
    units = np.clip(
        (RNG.poisson(14, n) * seasonal + marketing_spend / 300).round(), 1, None
    ).astype(int)
    unit_price = np.clip(RNG.normal(48, 14, n), 8, None).round(2)

    df = pd.DataFrame(
        {
            "order_id": [f"ORD{i:06d}" for i in range(1, n + 1)],
            "order_date": dates,
            "region": RNG.choice(["North", "South", "East", "West"], n),
            "channel": RNG.choice(["Online", "Retail", "Partner"], n, p=[0.55, 0.30, 0.15]),
            "units": units,
            "unit_price": unit_price,
            "revenue": (units * unit_price).round(2),
            "marketing_spend": marketing_spend,
            "customer_rating": np.clip(RNG.normal(4.1, 0.7, n), 1, 5).round(1),
        }
    )

    return _add_realistic_mess(df)


def _add_realistic_mess(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce the flaws a real export would have."""
    df = df.copy()
    n = len(df)

    # Missing values in two columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols[-2:]:
        gaps = RNG.choice(n, size=int(n * 0.06), replace=False)
        df.loc[gaps, col] = np.nan

    # A column that's almost entirely empty — the cleaner should drop it
    notes = pd.Series([None] * n, dtype="object")
    notes.iloc[RNG.choice(n, size=int(n * 0.03), replace=False)] = "check"
    df["legacy_notes"] = notes

    # Extreme outliers — cast to float first so integer columns can hold them
    outlier_col = numeric_cols[0]
    df[outlier_col] = df[outlier_col].astype(float)
    spikes = RNG.choice(n, size=max(3, int(n * 0.008)), replace=False)
    df.loc[spikes, outlier_col] = df[outlier_col].max() * 4.5

    # Duplicate rows, as if the export ran twice for some records
    duplicates = df.sample(int(n * 0.02), random_state=7)
    df = pd.concat([df, duplicates], ignore_index=True)

    return df.sample(frac=1, random_state=11).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = {"employees.csv": employees(), "sales.csv": sales()}

    for name, df in datasets.items():
        path = OUT_DIR / name
        df.to_csv(path, index=False)
        print(f"{path.relative_to(OUT_DIR.parent)}  —  {len(df):,} rows x {len(df.columns)} cols")

    # One Excel file, to exercise that code path
    excel_path = OUT_DIR / "employees.xlsx"
    datasets["employees.csv"].to_excel(excel_path, index=False)
    print(f"{excel_path.relative_to(OUT_DIR.parent)}  —  same data, Excel format")


if __name__ == "__main__":
    main()
