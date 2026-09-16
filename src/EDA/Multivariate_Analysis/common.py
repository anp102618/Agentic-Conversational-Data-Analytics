from __future__ import annotations

from typing import Any
import pandas as pd


def detect_column_type(series: pd.Series, datetime_threshold: float = 0.8) -> str:
    """Detect numeric, categorical, or datetime columns."""
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_object_dtype(series):
        converted = pd.to_datetime(series, errors="coerce")
        if len(series) > 0 and converted.notna().mean() >= datetime_threshold:
            return "datetime"
    return "categorical"


def get_column_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"numeric": [], "categorical": [], "datetime": []}
    for col in df.columns:
        groups[detect_column_type(df[col])].append(col)
    return groups


def dataset_metadata(df: pd.DataFrame) -> dict[str, Any]:
    groups = get_column_groups(df)
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": groups["numeric"],
        "categorical_columns": groups["categorical"],
        "datetime_columns": groups["datetime"],
    }