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
MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_ROWS = 1_000_000

#: Formats people reach for that we deliberately don't support. Each gets a
#: message saying what to do instead, because "unsupported format" alone
#: leaves someone stuck with a file they can't use.
CONVERSION_HINTS = {
    ".numbers": "Numbers files can't be read directly. In Numbers, use File → Export To → CSV.",
    ".gsheet": "Google Sheets shortcuts hold no data. In Sheets, use File → Download → CSV.",
    ".json": "JSON isn't tabular. Convert it to CSV first.",
    ".txt": "Plain text has no defined structure. If it's delimited, rename it to .csv or .tsv.",
    ".pdf": "PDFs aren't spreadsheets. Export the table to CSV from wherever it came from.",
    ".ods": "OpenDocument spreadsheets aren't supported. Save as .xlsx or .csv instead.",
    ".parquet": "Parquet isn't supported yet. Export to CSV.",
    ".zip": "Archives aren't unpacked. Upload the spreadsheet inside it directly.",
}


class IngestionError(ValueError):
    """The file can't be read as tabular data. Message is safe to show a user."""


def read_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded bytes into a dataframe."""
    suffix = Path(filename).suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        accepted = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        hint = CONVERSION_HINTS.get(suffix)

        if hint:
            raise IngestionError(f"{hint} Accepted formats: {accepted}.")

        raise IngestionError(
            f"'{suffix or filename}' isn't a supported file type. "
            f"Upload a spreadsheet in one of these formats: {accepted}."
        )

    if len(content) > MAX_FILE_BYTES:
        # One decimal place, because "0 MB" for an 11 MB limit reads as a bug
        raise IngestionError(
            f"That file is {len(content) / 1024 / 1024:.1f} MB and the limit is "
            f"{MAX_FILE_BYTES // 1024 // 1024} MB. Try filtering it down or "
            f"uploading a sample of the rows."
        )

    if not content:
        raise IngestionError("That file is empty — there's nothing to analyse.")

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
