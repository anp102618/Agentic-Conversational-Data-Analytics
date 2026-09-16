from __future__ import annotations

from typing import Any

import pandas as pd


def detect_column_type(
    s: pd.Series,
    datetime_threshold: float = 0.8
) -> str:
    """Return numeric, categorical, or datetime."""

    if pd.api.types.is_numeric_dtype(s):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    # Detect date-like string/object columns
    if pd.api.types.is_object_dtype(s):
        converted = pd.to_datetime(
            s,
            errors="coerce"
        )

        if len(s) and converted.notna().mean() >= datetime_threshold:
            return "datetime"

    return "categorical"


def safe_float(value: Any):
    return None if pd.isna(value) else float(value)


def column_metadata(s: pd.Series) -> dict:
    return {
        "total": int(len(s)),
        "missing": int(s.isna().sum()),
        "missing_percentage": float(
            s.isna().mean() * 100
        ),
        "non_missing": int(s.notna().sum()),
        "unique": int(s.nunique(dropna=True))
    }
