from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd


def detect_column_type(s: pd.Series, datetime_threshold: float = 0.8) -> str:
    """Detect whether a column is numeric, datetime, or categorical."""
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_object_dtype(s):
        converted = pd.to_datetime(s, errors="coerce")
        if len(s) > 0 and converted.notna().mean() >= datetime_threshold:
            return "datetime"
    return "categorical"


def pair_id(column1: str, column2: str) -> str:
    """Generate a stable identifier for a column pair."""
    return f"{column1}__{column2}"


def generate_pairs(df: pd.DataFrame):
    """Generate every unique pair of columns from the dataframe."""
    return combinations(df.columns, 2)


def pair_metadata(df: pd.DataFrame, column1: str, column2: str) -> dict[str, Any]:
    """Return metadata dictionary including types for a pair of columns."""
    return {
        "column_1": column1,
        "column_2": column2,
        "type_1": detect_column_type(df[column1]),
        "type_2": detect_column_type(df[column2]),
    }