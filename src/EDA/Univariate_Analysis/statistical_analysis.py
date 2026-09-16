from __future__ import annotations

import json
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.EDA.Univariate_Analysis.common import detect_column_type, safe_float, column_metadata
from src.config import BASE_DIR, config



def analyze_numeric(s: pd.Series) -> dict[str, Any]:
    """Perform descriptive, percentile, outlier, and normality analysis on a numeric series."""
    x = pd.to_numeric(s, errors="coerce").dropna()

    if len(x) < 2:
        return {
            "type": "numeric",
            "statistics": column_metadata(s),
            "error": "Not enough numeric observations."
        }

    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = x[(x < lower) | (x > upper)]
    
    skewness = stats.skew(x, bias=False)
    kurtosis = stats.kurtosis(x, bias=False)

    result = {
        "type": "numeric",
        "statistics": column_metadata(s),
        "descriptive": {
            "count": int(len(x)),
            "min": safe_float(x.min()),
            "max": safe_float(x.max()),
            "range": safe_float(x.max() - x.min()),
            "mean": safe_float(x.mean()),
            "median": safe_float(x.median()),
            "mode": safe_float(x.mode().iloc[0]) if not x.mode().empty else None,
            "std": safe_float(x.std()),
            "variance": safe_float(x.var()),
            "q1": safe_float(q1),
            "q2": safe_float(x.median()),
            "q3": safe_float(q3),
            "iqr": safe_float(iqr),
            "skewness": safe_float(skewness),
            "kurtosis": safe_float(kurtosis)
        },
        "percentiles": {
            "1%": safe_float(x.quantile(0.01)),
            "5%": safe_float(x.quantile(0.05)),
            "10%": safe_float(x.quantile(0.10)),
            "25%": safe_float(x.quantile(0.25)),
            "50%": safe_float(x.quantile(0.50)),
            "75%": safe_float(x.quantile(0.75)),
            "90%": safe_float(x.quantile(0.90)),
            "95%": safe_float(x.quantile(0.95)),
            "99%": safe_float(x.quantile(0.99))
        },
        "outliers": {
            "method": "IQR",
            "lower_fence": safe_float(lower),
            "upper_fence": safe_float(upper),
            "count": int(len(outliers)),
            "percentage": float(len(outliers) / len(x) * 100)
        }
    }

    if 3 <= len(x) <= 5000:
        try:
            statistic, p_value = stats.shapiro(x)
            result["normality"] = {
                "test": "Shapiro-Wilk",
                "statistic": safe_float(statistic),
                "p_value": safe_float(p_value),
                "normal_at_0.05": bool(p_value >= 0.05)
            }
        except Exception:
            result["normality"] = None

    return result


def analyze_categorical(s: pd.Series, top_n: int = 10) -> dict[str, Any]:
    """Perform frequency, mode, and entropy analysis on a categorical series."""
    x = s.dropna()
    counts = x.value_counts()

    result = {
        "type": "categorical",
        "statistics": column_metadata(s),
        "descriptive": {
            "count": int(len(x)),
            "unique": int(x.nunique()),
            "mode": str(counts.index[0]) if len(counts) else None,
            "mode_count": int(counts.iloc[0]) if len(counts) else 0,
            "mode_percentage": float(counts.iloc[0] / len(x) * 100) if len(x) else 0
        },
        "frequencies": [
            {
                "category": str(cat),
                "count": int(cnt),
                "percentage": float(cnt / len(x) * 100)
            }
            for cat, cnt in counts.head(top_n).items()
        ]
    }

    if len(x):
        probabilities = counts / len(x)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        result["descriptive"]["entropy"] = safe_float(entropy)

    return result


def analyze_datetime(s: pd.Series) -> dict[str, Any]:
    """Perform temporal range, component breakdown, and gap analysis on a datetime series."""
    x = pd.to_datetime(s, errors="coerce").dropna()

    if len(x) == 0:
        return {
            "type": "datetime",
            "statistics": column_metadata(s),
            "error": "No valid datetime observations."
        }

    minimum, maximum = x.min(), x.max()
    duration = maximum - minimum

    result = {
        "type": "datetime",
        "statistics": column_metadata(s),
        "descriptive": {
            "count": int(len(x)),
            "min": minimum.isoformat(),
            "max": maximum.isoformat(),
            "range_days": float(duration.total_seconds() / 86400),
            "range_hours": float(duration.total_seconds() / 3600),
            "unique_dates": int(x.dt.date.nunique()),
            "unique_timestamps": int(x.nunique())
        },
        "components": {
            "years": sorted(x.dt.year.unique().tolist()),
            "months": sorted(x.dt.month.unique().tolist()),
            "days_of_week": sorted(x.dt.dayofweek.unique().tolist()),
            "hours": sorted(x.dt.hour.unique().tolist())
        },
        "frequency": {
            "year": {str(k): int(v) for k, v in x.dt.year.value_counts().sort_index().items()},
            "month": {str(k): int(v) for k, v in x.dt.month.value_counts().sort_index().items()},
            "day_of_week": {str(k): int(v) for k, v in x.dt.dayofweek.value_counts().sort_index().items()}
        }
    }

    if len(x) > 1:
        gaps = x.sort_values().diff().dropna().dt.total_seconds()
        result["time_gaps"] = {
            "mean_hours": safe_float(gaps.mean() / 3600),
            "median_hours": safe_float(gaps.median() / 3600),
            "min_hours": safe_float(gaps.min() / 3600),
            "max_hours": safe_float(gaps.max() / 3600)
        }

    return result


def analyze_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Iterate through all columns in a DataFrame and perform appropriate type-based analyses."""
    result = {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(len(df.columns))
        },
        "variables": {}
    }

    for column in df.columns:
        s = df[column]
        col_type = detect_column_type(s)

        if col_type == "numeric":
            result["variables"][column] = analyze_numeric(s)
        elif col_type == "datetime":
            result["variables"][column] = analyze_datetime(s)
        else:
            result["variables"][column] = analyze_categorical(s)

    return result


def save_analysis(analysis: dict[str, Any], filename: Path = BASE_DIR / "src" /"EDA" / "Univariate_Analysis" / "Reports" / "statistical_analysis.json") -> None:
    """Save the analysis dictionary to a JSON file with UTF-8 encoding."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    df = pd.read_csv("data.csv")
    analysis = analyze_dataframe(df)
    save_analysis(analysis, filename=BASE_DIR / "src" / "EDA" / "Univariate_Analysis" / "Reports" / "statistical_analysis.json")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    