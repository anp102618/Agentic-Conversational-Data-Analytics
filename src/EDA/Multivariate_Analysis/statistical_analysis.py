from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.EDA.Multivariate_Analysis.common import dataset_metadata, get_column_groups
from src.config import BASE_DIR


# ============================================================
# Helpers
# ============================================================

def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


# ============================================================
# Numeric Multivariate Statistics
# ============================================================

def analyze_numeric_features(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = get_column_groups(df)["numeric"]
    if not numeric_columns:
        return {"available": False, "reason": "No numeric variables."}

    numeric = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    result = {"available": True, "variables": numeric_columns}

    result["pearson_correlation"] = numeric.corr(method="pearson").to_dict()
    result["spearman_correlation"] = numeric.corr(method="spearman").to_dict()
    result["kendall_correlation"] = numeric.corr(method="kendall").to_dict()
    result["covariance_matrix"] = numeric.cov().to_dict()

    relationships = []
    for i, col1 in enumerate(numeric_columns):
        for col2 in numeric_columns[i + 1:]:
            paired = numeric[[col1, col2]].dropna()
            if len(paired) < 3:
                continue
            x, y = paired[col1], paired[col2]
            r, p = stats.pearsonr(x, y)
            rho, rho_p = stats.spearmanr(x, y)
            relationships.append({
                "variable_1": col1,
                "variable_2": col2,
                "sample_size": int(len(paired)),
                "pearson": {"coefficient": safe_float(r), "p_value": safe_float(p)},
                "spearman": {"coefficient": safe_float(rho), "p_value": safe_float(rho_p)}
            })

    result["pairwise_relationships"] = relationships
    return result


# ============================================================
# Multicollinearity
# ============================================================

def calculate_vif(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = get_column_groups(df)["numeric"]
    if len(numeric_columns) < 2:
        return {"available": False, "reason": "At least two numeric variables required."}

    data = df[numeric_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        return {"available": False, "reason": "Not enough complete observations."}

    result = []
    for col in numeric_columns:
        y = data[col]
        others = [c for c in numeric_columns if c != col]
        if not others:
            continue
        X = data[others]
        try:
            X_mat = np.column_stack([np.ones(len(X)), X.values])
            coeffs = np.linalg.lstsq(X_mat, y.values, rcond=None)[0]
            preds = X_mat @ coeffs
            ss_res = np.sum((y.values - preds) ** 2)
            ss_tot = np.sum((y.values - y.mean()) ** 2)
            if ss_tot == 0:
                continue
            r_squared = 1 - ss_res / ss_tot
            vif = float("inf") if r_squared >= 0.999999 else 1 / (1 - r_squared)
            result.append({
                "variable": col,
                "r_squared": safe_float(r_squared),
                "vif": None if np.isinf(vif) else safe_float(vif)
            })
        except Exception:
            continue

    return {"available": True, "variables": result}


# ============================================================
# PCA
# ============================================================

def analyze_pca(df: pd.DataFrame) -> dict[str, Any]:
    numeric_columns = get_column_groups(df)["numeric"]
    if len(numeric_columns) < 2:
        return {"available": False, "reason": "At least two numeric variables required."}

    data = df[numeric_columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        return {"available": False, "reason": "Not enough complete observations."}

    X = data.values
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=1)
    stds[stds == 0] = 1
    X_scaled = (X - means) / stds

    _, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    eigenvalues = (S ** 2) / (len(X_scaled) - 1)
    explained_variance = eigenvalues / eigenvalues.sum()
    cumulative = np.cumsum(explained_variance)

    loadings = {
        f"PC{i + 1}": {var: safe_float(comp[j]) for j, var in enumerate(numeric_columns)}
        for i, comp in enumerate(Vt)
    }

    return {
        "available": True,
        "sample_size": int(len(data)),
        "variables": numeric_columns,
        "n_components": int(len(eigenvalues)),
        "eigenvalues": [safe_float(v) for v in eigenvalues],
        "explained_variance_ratio": [safe_float(v) for v in explained_variance],
        "cumulative_explained_variance": [safe_float(v) for v in cumulative],
        "loadings": loadings
    }


# ============================================================
# Categorical Association
# ============================================================

def analyze_categorical_features(df: pd.DataFrame) -> dict[str, Any]:
    categorical_columns = get_column_groups(df)["categorical"]
    if len(categorical_columns) < 2:
        return {"available": False, "reason": "At least two categorical variables required."}

    relationships = []
    for i, col1 in enumerate(categorical_columns):
        for col2 in categorical_columns[i + 1:]:
            data = df[[col1, col2]].dropna()
            if data.empty:
                continue
            table = pd.crosstab(data[col1], data[col2])
            if table.shape[0] < 2 or table.shape[1] < 2:
                continue
            try:
                chi2, p, dof, _ = stats.chi2_contingency(table)
                n = table.values.sum()
                r, k = table.shape
                cramers_v = np.sqrt(chi2 / (n * min(k - 1, r - 1)))
                relationships.append({
                    "variable_1": col1,
                    "variable_2": col2,
                    "sample_size": int(n),
                    "chi_square": safe_float(chi2),
                    "p_value": safe_float(p),
                    "degrees_of_freedom": int(dof),
                    "cramers_v": safe_float(cramers_v)
                })
            except Exception:
                continue

    return {"available": True, "variables": categorical_columns, "pairwise_associations": relationships}


# ============================================================
# Grouped Numeric Analysis
# ============================================================

def analyze_grouped_numeric(df: pd.DataFrame, max_categories: int = 20) -> dict[str, Any]:
    groups = get_column_groups(df)
    numeric_columns, categorical_columns = groups["numeric"], groups["categorical"]
    if not numeric_columns or not categorical_columns:
        return {"available": False, "reason": "Both numeric and categorical variables are required."}

    analyses = []
    for num_col in numeric_columns:
        for cat_col in categorical_columns:
            data = df[[num_col, cat_col]].copy()
            data[num_col] = pd.to_numeric(data[num_col], errors="coerce")
            data = data.dropna()

            top_cats = data[cat_col].value_counts().head(max_categories).index
            data = data[data[cat_col].isin(top_cats)]

            grouped = data.groupby(cat_col, observed=True)[num_col].agg(["count", "mean", "median", "std", "min", "max"])
            analyses.append({
                "numeric_variable": num_col,
                "grouping_variable": cat_col,
                "groups": {
                    str(idx): {k: (int(v) if k == "count" else safe_float(v)) for k, v in row.items()}
                    for idx, row in grouped.iterrows()
                }
            })

    return {"available": True, "analyses": analyses}


# ============================================================
# Temporal Multivariate Analysis
# ============================================================

def analyze_temporal_features(df: pd.DataFrame) -> dict[str, Any]:
    groups = get_column_groups(df)
    datetime_columns, numeric_columns = groups["datetime"], groups["numeric"]
    if not datetime_columns or not numeric_columns:
        return {"available": False, "reason": "Both datetime and numeric variables are required."}

    analyses = []
    for dt_col in datetime_columns:
        data = df.copy()
        data[dt_col] = pd.to_datetime(data[dt_col], errors="coerce")
        data = data.dropna(subset=[dt_col]).set_index(dt_col)
        if data.empty:
            continue

        numeric_data = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
        monthly = numeric_data.resample("ME").mean()

        analyses.append({
            "datetime_variable": dt_col,
            "frequency": "monthly",
            "numeric_variables": numeric_columns,
            "period_count": int(len(monthly)),
            "summary": {
                str(idx): {col: safe_float(val) for col, val in row.items()}
                for idx, row in monthly.iterrows()
            }
        })

    return {"available": True, "analyses": analyses}


# ============================================================
# Complete Multivariate Analysis
# ============================================================

def analyze_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "analysis_type": "multivariate",
        "dataset": dataset_metadata(df),
        "numeric_analysis": analyze_numeric_features(df),
        "multicollinearity": calculate_vif(df),
        "pca": analyze_pca(df),
        "categorical_analysis": analyze_categorical_features(df),
        "grouped_numeric_analysis": analyze_grouped_numeric(df),
        "temporal_analysis": analyze_temporal_features(df)
    }


def save_analysis(analysis: dict[str, Any], filename: Path = BASE_DIR / "src" / "EDA" / "Multivariate_Analysis" / "Reports" / "multivariate_statistics.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    analysis = analyze_dataframe(df)
    save_analysis(analysis, filename=BASE_DIR / "src" / "EDA" / "Multivariate_Analysis" / "Reports" / "multivariate_statistics.json")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))