"""
Reading uploaded files into dataframes, with the validation that stops
a malformed upload from becoming a confusing error three agents later.
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB
MAX_ROWS = 1_000_000


class IngestionError(ValueError):
    """The file can't be read as tabular data. Message is safe to show a user."""


def read_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded bytes into a dataframe."""
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"'{suffix or filename}' isn't a supported format. "
            f"Upload one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if len(content) > MAX_FILE_BYTES:
        raise IngestionError(
            f"File is {len(content) / 1024 / 1024:.0f} MB. The limit is "
            f"{MAX_FILE_BYTES // 1024 // 1024} MB."
        )

    if not content:
        raise IngestionError("The file is empty.")

    try:
        if suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(BytesIO(content))
        else:
            separator = "\t" if suffix == ".tsv" else ","
            df = _read_csv_with_fallback(content, separator)
    except IngestionError:
        raise
    except Exception as exc:
        raise IngestionError(f"Could not read the file: {exc}") from exc

    return _validate(df, filename)


def read_path(path: str | Path) -> pd.DataFrame:
    """Read a file from disk. Used by the CLI and tests."""
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"No file at {path}")
    return read_bytes(path.read_bytes(), path.name)


def _read_csv_with_fallback(content: bytes, separator: str) -> pd.DataFrame:
    """Try UTF-8, then fall back to encodings real-world CSVs actually use."""
    last_error: Exception | None = None

    for encoding in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return pd.read_csv(BytesIO(content), sep=separator, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        except pd.errors.EmptyDataError as exc:
            raise IngestionError("The file has no rows.") from exc

    raise IngestionError(f"Could not decode the file's text encoding: {last_error}")


def _validate(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    if df.empty:
        raise IngestionError(f"{filename} parsed successfully but contains no rows.")

    if len(df.columns) < 2:
        raise IngestionError(
            f"{filename} has only {len(df.columns)} column. "
            "Analysis needs at least two to find relationships."
        )

    if len(df) > MAX_ROWS:
        log.warning("Sampling %s from %d rows down to %d", filename, len(df), MAX_ROWS)
        df = df.sample(MAX_ROWS, random_state=42).reset_index(drop=True)

    # Unnamed columns come from stray index columns in exported CSVs
    df.columns = [
        str(c).strip() if not str(c).startswith("Unnamed:") else f"column_{i}"
        for i, c in enumerate(df.columns)
    ]

    return df
