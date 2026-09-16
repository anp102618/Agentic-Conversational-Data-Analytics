from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# BASIC SAFE CONVERSION HELPERS
# ============================================================

def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        res = float(value)
        return res if np.isfinite(res) else None
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# TARGET TYPE DETECTION
# ============================================================

def infer_target_type(y: pd.Series, classification_threshold: int = 10) -> str:
    if not isinstance(y, pd.Series):
        raise TypeError("Target must be provided as a pandas Series.")
    
    y_clean = y.dropna()
    if y_clean.empty:
        raise ValueError("Target contains no non-missing observations.")
    if pd.api.types.is_datetime64_any_dtype(y_clean):
        raise ValueError("Datetime targets are not supported.")
    
    if (
        pd.api.types.is_bool_dtype(y_clean)
        or pd.api.types.is_object_dtype(y_clean)
        or pd.api.types.is_string_dtype(y_clean)
        or isinstance(y_clean.dtype, pd.CategoricalDtype)
    ):
        return "classification"
    
    if pd.api.types.is_numeric_dtype(y_clean):
        return "classification" if y_clean.nunique() <= classification_threshold else "regression"
    
    return "classification"


# ============================================================
# FEATURE TYPE DETECTION
# ============================================================

def get_feature_types(df: pd.DataFrame) -> dict[str, list[str]]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    
    return {
        "numeric": df.select_dtypes(include=np.number).columns.tolist(),
        "categorical": df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist(),
        "datetime": df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist(),
    }


# ============================================================
# DATASET + TARGET METADATA
# ============================================================

def dataset_target_metadata(df: pd.DataFrame, target: str, target_type: str) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' was not found.")
    
    y = df[target]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "target": str(target),
        "target_type": str(target_type),
        "missing_target": int(y.isna().sum()),
        "missing_target_pct": safe_float(y.isna().mean() * 100),
        "unique_target_values": int(y.nunique(dropna=True)),
        "feature_count": int(len(df.columns) - 1),
    }


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    
    return value


# ============================================================
# FEATURE NAME CLEANING & MAPPING
# ============================================================

def clean_feature_name(name: str) -> str:
    name = str(name)
    for old in ("numeric__", "categorical__", "onehot__"):
        name = name.replace(old, "")
    return name


def original_feature_name(transformed_name: str, original_columns: list[str]) -> str:
    cleaned = clean_feature_name(transformed_name)
    sorted_cols = sorted(original_columns, key=len, reverse=True)
    
    for col in sorted_cols:
        col_str = str(col)
        if cleaned == col_str or any(cleaned.startswith(col_str + s) for s in ("_", "=", "[")):
            return col_str
            
    return cleaned