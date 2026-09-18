from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from src.config import BASE_DIR, config

from src.EDA.Bivariate_Analysis.common import (
    detect_column_type,
    generate_pairs,
    pair_id,
    pair_metadata,
)

# Maximum unique values allowed for categorical bivariate analysis to prevent MemoryError
MAX_CATEGORICAL_CARDINALITY = 200


def safe_float(value: Any) -> float | None:
    """Convert a value to a JSON-safe float."""
    return None if pd.isna(value) else float(value)


def safe_pvalue(value: Any) -> float | None:
    """Convert a p-value to a JSON-safe float."""
    return None if pd.isna(value) else float(value)


def analyze_numeric_numeric(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Analyze bivariate relationships between two numeric series."""
    data = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
        }
    ).dropna()

    result = {"pair_type": "numeric_numeric", "sample_size": int(len(data))}
    if len(data) < 3:
        result["error"] = "Not enough paired numeric observations."
        return result

    xv, yv = data["x"].values, data["y"].values
    pearson_r, pearson_p = stats.pearsonr(xv, yv)
    spearman_rho, spearman_p = stats.spearmanr(xv, yv)
    kendall_tau, kendall_p = stats.kendalltau(xv, yv)
    covariance = np.cov(xv, yv, ddof=1)[0, 1]

    regression = None
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(xv, yv)
        regression = {
            "slope": safe_float(slope),
            "intercept": safe_float(intercept),
            "r_squared": safe_float(r_value**2),
            "p_value": safe_pvalue(p_value),
            "standard_error": safe_float(std_err),
        }
    except Exception:
        pass

    result["correlation"] = {
        "pearson": {
            "coefficient": safe_float(pearson_r),
            "p_value": safe_pvalue(pearson_p),
            "significant_at_0.05": bool(pearson_p < 0.05),
        },
        "spearman": {
            "coefficient": safe_float(spearman_rho),
            "p_value": safe_pvalue(spearman_p),
            "significant_at_0.05": bool(spearman_p < 0.05),
        },
        "kendall": {
            "coefficient": safe_float(kendall_tau),
            "p_value": safe_pvalue(kendall_p),
            "significant_at_0.05": bool(kendall_p < 0.05),
        },
        "covariance": safe_float(covariance),
    }
    result["linear_regression"] = regression

    abs_r = abs(pearson_r)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very_strong"

    result["interpretation"] = {
        "pearson_direction": (
            "positive"
            if pearson_r > 0
            else "negative"
            if pearson_r < 0
            else "none"
        ),
        "pearson_strength": strength,
    }
    return result


def analyze_numeric_categorical(
    numeric: pd.Series, categorical: pd.Series
) -> dict[str, Any]:
    """Analyze relationships between a numeric and a categorical series."""
    if categorical.nunique() > MAX_CATEGORICAL_CARDINALITY:
        return {
            "pair_type": "numeric_categorical",
            "sample_size": int(len(numeric.dropna())),
            "error": f"Skipped: Categorical column exceeds max cardinality threshold ({MAX_CATEGORICAL_CARDINALITY})."
        }

    data = pd.DataFrame(
        {
            "numeric": pd.to_numeric(numeric, errors="coerce"),
            "category": categorical,
        }
    ).dropna()

    result = {
        "pair_type": "numeric_categorical",
        "sample_size": int(len(data)),
        "groups": int(data["category"].nunique()),
    }
    if len(data) == 0:
        result["error"] = "No valid paired observations."
        return result

    grouped = data.groupby("category", observed=True)["numeric"]
    group_statistics = [
        {
            "category": str(cat),
            "count": int(vals.count()),
            "mean": safe_float(vals.mean()),
            "median": safe_float(vals.median()),
            "std": safe_float(vals.std()),
            "min": safe_float(vals.min()),
            "max": safe_float(vals.max()),
            "q1": safe_float(vals.quantile(0.25)),
            "q3": safe_float(vals.quantile(0.75)),
        }
        for cat, vals in grouped
    ]
    result["group_statistics"] = group_statistics

    groups = [
        vals.dropna().values for _, vals in grouped if len(vals.dropna()) >= 2
    ]
    result["tests"] = {}

    if len(groups) == 2:
        g1, g2 = groups
        try:
            stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
            result["tests"]["welch_t_test"] = {
                "statistic": safe_float(stat),
                "p_value": safe_pvalue(p_val),
                "significant_at_0.05": bool(p_val < 0.05),
            }
        except Exception:
            pass
        try:
            stat, p_val = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            result["tests"]["mann_whitney_u"] = {
                "statistic": safe_float(stat),
                "p_value": safe_pvalue(p_val),
                "significant_at_0.05": bool(p_val < 0.05),
            }
        except Exception:
            pass
    elif len(groups) >= 3:
        try:
            stat, p_val = stats.f_oneway(*groups)
            result["tests"]["one_way_anova"] = {
                "statistic": safe_float(stat),
                "p_value": safe_pvalue(p_val),
                "significant_at_0.05": bool(p_val < 0.05),
            }
        except Exception:
            pass
        try:
            stat, p_val = stats.kruskal(*groups)
            result["tests"]["kruskal_wallis"] = {
                "statistic": safe_float(stat),
                "p_value": safe_pvalue(p_val),
                "significant_at_0.05": bool(p_val < 0.05),
            }
        except Exception:
            pass

    overall_std = data["numeric"].std()
    if overall_std and not pd.isna(overall_std):
        means = data.groupby("category", observed=True)["numeric"].mean()
        result["effect"] = {
            "mean_range": safe_float(means.max() - means.min()),
            "standardized_mean_range": safe_float(
                (means.max() - means.min()) / overall_std
            ),
        }

    return result


def analyze_categorical_categorical(
    x: pd.Series, y: pd.Series
) -> dict[str, Any]:
    """Analyze relationships between two categorical series."""
    if x.nunique() > MAX_CATEGORICAL_CARDINALITY or y.nunique() > MAX_CATEGORICAL_CARDINALITY:
        return {
            "pair_type": "categorical_categorical",
            "sample_size": int(len(x.dropna())),
            "error": f"Skipped: One or both categorical columns exceed max cardinality threshold ({MAX_CATEGORICAL_CARDINALITY})."
        }

    data = pd.DataFrame({"x": x, "y": y}).dropna()
    result = {"pair_type": "categorical_categorical", "sample_size": int(len(data))}
    if len(data) == 0:
        result["error"] = "No valid paired observations."
        return result

    table = pd.crosstab(data["x"], data["y"])
    result["contingency_table"] = {
        "rows": [str(r) for r in table.index],
        "columns": [str(c) for c in table.columns],
        "values": table.values.tolist(),
    }

    row_percentages = table.div(table.sum(axis=1), axis=0) * 100
    result["row_percentages"] = {
        str(idx): {str(col): safe_float(val) for col, val in row.items()}
        for idx, row in row_percentages.iterrows()
    }

    if table.shape[0] >= 2 and table.shape[1] >= 2:
        try:
            chi2, p_val, dof, _ = stats.chi2_contingency(table)
            result["chi_square"] = {
                "statistic": safe_float(chi2),
                "p_value": safe_pvalue(p_val),
                "degrees_of_freedom": int(dof),
                "significant_at_0.05": bool(p_val < 0.05),
            }
            n = table.values.sum()
            r_dim, k_dim = table.shape
            if n > 0:
                cramers_v = np.sqrt(chi2 / (n * min(k_dim - 1, r_dim - 1)))
                result["effect_size"] = {"cramers_v": safe_float(cramers_v)}
        except Exception:
            pass

    return result


def analyze_datetime_numeric(
    datetime_series: pd.Series, numeric_series: pd.Series
) -> dict[str, Any]:
    """Analyze temporal trends between a datetime and a numeric series."""
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(datetime_series, errors="coerce"),
            "value": pd.to_numeric(numeric_series, errors="coerce"),
        }
    ).dropna()

    result = {"pair_type": "datetime_numeric", "sample_size": int(len(data))}
    if len(data) < 3:
        result["error"] = "Not enough paired observations."
        return result

    data = data.sort_values("date")
    result["time_range"] = {
        "start": data["date"].min().isoformat(),
        "end": data["date"].max().isoformat(),
    }

    elapsed_days = (data["date"] - data["date"].min()).dt.total_seconds() / 86400
    pearson_r, pearson_p = stats.pearsonr(elapsed_days, data["value"])
    spearman_rho, spearman_p = stats.spearmanr(elapsed_days, data["value"])

    result["trend"] = {
        "pearson": {
            "coefficient": safe_float(pearson_r),
            "p_value": safe_pvalue(pearson_p),
        },
        "spearman": {
            "coefficient": safe_float(spearman_rho),
            "p_value": safe_pvalue(spearman_p),
        },
    }

    slope, intercept, r, p, stderr = stats.linregress(elapsed_days, data["value"])
    result["linear_trend"] = {
        "slope_per_day": safe_float(slope),
        "intercept": safe_float(intercept),
        "r_squared": safe_float(r**2),
        "p_value": safe_pvalue(p),
        "standard_error": safe_float(stderr),
    }

    daily = (
        data.set_index("date")["value"]
        .resample("D")
        .agg(["count", "mean", "median", "std"])
        .dropna(how="all")
    )
    result["daily_summary"] = {
        "days": int(len(daily)),
        "mean": safe_float(daily["mean"].mean()),
        "median": safe_float(daily["median"].median()),
        "min_mean": safe_float(daily["mean"].min()),
        "max_mean": safe_float(daily["mean"].max()),
    }
    return result


def analyze_datetime_categorical(
    datetime_series: pd.Series, categorical_series: pd.Series
) -> dict[str, Any]:
    """Analyze frequencies of a categorical series over a datetime series."""
    if categorical_series.nunique() > MAX_CATEGORICAL_CARDINALITY:
        return {
            "pair_type": "datetime_categorical",
            "sample_size": int(len(datetime_series.dropna())),
            "error": f"Skipped: Categorical column exceeds max cardinality threshold ({MAX_CATEGORICAL_CARDINALITY})."
        }

    data = pd.DataFrame(
        {
            "date": pd.to_datetime(datetime_series, errors="coerce"),
            "category": categorical_series,
        }
    ).dropna()

    result = {"pair_type": "datetime_categorical", "sample_size": int(len(data))}
    if len(data) == 0:
        result["error"] = "No valid paired observations."
        return result

    counts = pd.crosstab(data["date"].dt.to_period("M"), data["category"])
    result["monthly_counts"] = {
        str(idx): {str(col): int(val) for col, val in row.items()}
        for idx, row in counts.iterrows()
    }

    totals = data["category"].value_counts()
    result["category_totals"] = {str(cat): int(cnt) for cat, cnt in totals.items()}
    return result


def analyze_datetime_datetime(x: pd.Series, y: pd.Series) -> dict[str, Any]:
    """Analyze time gaps and correlations between two datetime series."""
    data = pd.DataFrame(
        {
            "x": pd.to_datetime(x, errors="coerce"),
            "y": pd.to_datetime(y, errors="coerce"),
        }
    ).dropna()

    result = {"pair_type": "datetime_datetime", "sample_size": int(len(data))}
    if len(data) == 0:
        result["error"] = "No valid paired observations."
        return result

    gap = (data["y"] - data["x"]).dt.total_seconds()
    result["time_difference"] = {
        "mean_hours": safe_float(gap.mean() / 3600),
        "median_hours": safe_float(gap.median() / 3600),
        "min_hours": safe_float(gap.min() / 3600),
        "max_hours": safe_float(gap.max() / 3600),
        "negative_count": int((gap < 0).sum()),
        "zero_count": int((gap == 0).sum()),
        "positive_count": int((gap > 0).sum()),
    }

    if len(data) >= 3:
        x_days = (data["x"] - data["x"].min()).dt.total_seconds() / 86400
        y_days = (data["y"] - data["y"].min()).dt.total_seconds() / 86400
        r, p = stats.pearsonr(x_days, y_days)
        result["correlation"] = {
            "pearson": {"coefficient": safe_float(r), "p_value": safe_pvalue(p)}
        }

    return result


def analyze_pair(df: pd.DataFrame, column1: str, column2: str) -> dict[str, Any]:
    """Route a column pair to the correct bivariate analysis function."""
    type1, type2 = detect_column_type(df[column1]), detect_column_type(df[column2])
    metadata = pair_metadata(df, column1, column2)

    if type1 == "numeric" and type2 == "numeric":
        analysis = analyze_numeric_numeric(df[column1], df[column2])
    elif type1 == "numeric" and type2 == "categorical":
        analysis = analyze_numeric_categorical(df[column1], df[column2])
    elif type1 == "categorical" and type2 == "numeric":
        analysis = analyze_numeric_categorical(df[column2], df[column1])
    elif type1 == "categorical" and type2 == "categorical":
        analysis = analyze_categorical_categorical(df[column1], df[column2])
    elif type1 == "datetime" and type2 == "numeric":
        analysis = analyze_datetime_numeric(df[column1], df[column2])
    elif type1 == "numeric" and type2 == "datetime":
        analysis = analyze_datetime_numeric(df[column2], df[column1])
    elif type1 == "datetime" and type2 == "categorical":
        analysis = analyze_datetime_categorical(df[column1], df[column2])
    elif type1 == "categorical" and type2 == "datetime":
        analysis = analyze_datetime_categorical(df[column2], df[column1])
    elif type1 == "datetime" and type2 == "datetime":
        analysis = analyze_datetime_datetime(df[column1], df[column2])
    else:
        analysis = {"pair_type": "unsupported"}

    return {
        "pair_id": pair_id(column1, column2),
        "variables": metadata,
        "analysis": analysis,
    }


def analyze_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Run bivariate analyses across all unique column pairs in a DataFrame."""
    result = {
        "dataset": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "pairs": {},
    }
    for column1, column2 in generate_pairs(df):
        pair = analyze_pair(df, column1, column2)
        result["pairs"][pair["pair_id"]] = pair
    return result


def save_analysis(analysis: dict[str, Any], filename: Path) -> None:
    """Save the analysis dictionary to a JSON file."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


# ==========================================
# IMPLEMENTATION 
# ==========================================

def perform_bivariate_statistical_analysis(file_path: Path) -> dict[str, Any]:
    """Perform bivariate statistical analysis on the DataFrame."""
    df = pd.read_csv(file_path)
    analysis = analyze_dataframe(df)
    save_analysis(analysis, filename=BASE_DIR / "src" / "EDA" / "Bivariate_Analysis" / "Reports" / "bivariate_statistics.json")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    perform_bivariate_statistical_analysis(file_path=BASE_DIR / "src" / "Data" / "ecommerce_sales.csv")