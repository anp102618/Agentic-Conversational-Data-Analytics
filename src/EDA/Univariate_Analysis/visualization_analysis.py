from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.EDA.Univariate_Analysis.common import detect_column_type
from src.config import BASE_DIR, config


# ============================================================
# Visualization specifications
# ============================================================

def numeric_visualizations(column: str, bins: int = 30) -> list[dict[str, Any]]:
    """Return metadata specifications for numeric column visualizations."""
    return [
        {"id": f"{column}_histogram", "type": "histogram", "purpose": "distribution", "x": column, "bins": bins},
        {"id": f"{column}_boxplot", "type": "boxplot", "purpose": "outlier_detection", "x": column},
        {"id": f"{column}_kde", "type": "kde", "purpose": "distribution_shape", "x": column},
        {"id": f"{column}_qq", "type": "qq_plot", "purpose": "normality_assessment", "x": column}
    ]


def categorical_visualizations(column: str) -> list[dict[str, Any]]:
    """Return metadata specifications for categorical column visualizations."""
    return [
        {"id": f"{column}_bar", "type": "bar", "purpose": "category_frequency", "x": column}
    ]


def datetime_visualizations(column: str) -> list[dict[str, Any]]:
    """Return metadata specifications for datetime column visualizations."""
    return [
        {"id": f"{column}_timeline", "type": "line", "purpose": "time_distribution", "x": column},
        {"id": f"{column}_year", "type": "bar", "purpose": "year_frequency", "x": column, "time_unit": "year"},
        {"id": f"{column}_month", "type": "bar", "purpose": "month_frequency", "x": column, "time_unit": "month"},
        {"id": f"{column}_weekday", "type": "bar", "purpose": "weekday_frequency", "x": column, "time_unit": "day_of_week"}
    ]


# ============================================================
# Build visualization metadata
# ============================================================

def analyze_visualizations(df: pd.DataFrame, bins: int = 30) -> dict[str, Any]:
    """Generate a complete dictionary of visualization metadata for all dataframe columns."""
    result = {
        "dataset": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "variables": {}
    }

    for column in df.columns:
        column_type = detect_column_type(df[column])
        
        if column_type == "numeric":
            visualizations = numeric_visualizations(column, bins)
        elif column_type == "datetime":
            visualizations = datetime_visualizations(column)
        else:
            visualizations = categorical_visualizations(column)

        result["variables"][column] = {
            "type": column_type,
            "visualizations": visualizations
        }

    return result


# ============================================================
# Save visualization JSON
# ============================================================

def save_visualization_analysis(analysis: dict[str, Any], filename: Path = BASE_DIR / "src" /"EDA" / "Univariate_Analysis" / "Reports" / "visualization_analysis.json") -> None:
    """Save visualization metadata dictionary to a JSON file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


# ============================================================
# Numeric rendering
# ============================================================

def render_numeric(df: pd.DataFrame, column: str, bins: int = 30) -> plt.Figure:
    """Render a 2x2 matplotlib figure containing numeric diagnostic plots (Histogram, Boxplot, KDE, QQ)."""
    x = pd.to_numeric(df[column], errors="coerce").dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Histogram
    axes[0, 0].hist(x, bins=bins, edgecolor="black", alpha=0.75)
    axes[0, 0].set(title=f"Histogram: {column}", xlabel=column, ylabel="Frequency")

    # Boxplot
    axes[0, 1].boxplot(x, vert=False)
    axes[0, 1].set(title=f"Boxplot: {column}", xlabel=column)

    # KDE
    if x.nunique() >= 5:
        kde = stats.gaussian_kde(x)
        grid = np.linspace(x.min(), x.max(), 500)
        density = kde(grid)
        axes[1, 0].plot(grid, density)
        axes[1, 0].fill_between(grid, density, alpha=0.2)

    axes[1, 0].set(title=f"KDE: {column}", xlabel=column, ylabel="Density")

    # QQ plot
    stats.probplot(x, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title(f"QQ Plot: {column}")

    plt.tight_layout()
    return fig


# ============================================================
# Categorical rendering
# ============================================================

def render_categorical(df: pd.DataFrame, column: str, top_n: int = 10) -> plt.Figure:
    """Render a horizontal bar chart showing category value counts."""
    counts = df[column].dropna().value_counts().head(top_n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    counts.plot(kind="barh", ax=ax)
    ax.set(title=f"Category Frequency: {column}", xlabel="Count", ylabel=column)

    plt.tight_layout()
    return fig


# ============================================================
# Datetime rendering
# ============================================================

def render_datetime(df: pd.DataFrame, column: str) -> plt.Figure:
    """Render a 2x2 matplotlib figure for datetime frequency patterns (Timeline, Year, Month, Weekday)."""
    x = pd.to_datetime(df[column], errors="coerce").dropna()
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Timeline
    timeline = x.dt.floor("D").value_counts().sort_index()
    axes[0, 0].plot(timeline.index, timeline.values)
    axes[0, 0].set(title=f"Timeline: {column}", xlabel="Date", ylabel="Count")

    # Year
    x.dt.year.value_counts().sort_index().plot(kind="bar", ax=axes[0, 1])
    axes[0, 1].set_title(f"Frequency by Year: {column}")

    # Month
    x.dt.month.value_counts().sort_index().plot(kind="bar", ax=axes[1, 0])
    axes[1, 0].set_title(f"Frequency by Month: {column}")

    # Weekday
    x.dt.dayofweek.value_counts().sort_index().plot(kind="bar", ax=axes[1, 1])
    axes[1, 1].set_title(f"Frequency by Day of Week: {column}")

    plt.tight_layout()
    return fig


# ============================================================
# Render all plots
# ============================================================

def render_visualizations(df: pd.DataFrame, output_dir: Path = "plots", bins: int = 30) -> None:
    """Iterate through dataframe columns, generate their respective visualization figures, and save as PNGs."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for column in df.columns:
        column_type = detect_column_type(df[column])

        if column_type == "numeric":
            fig = render_numeric(df, column, bins)
        elif column_type == "datetime":
            fig = render_datetime(df, column)
        else:
            fig = render_categorical(df, column)

        fig.savefig(output / f"{column}_visualization.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    # Create visualization metadata
    visualization_analysis = analyze_visualizations(df, bins=30)

    # Save JSON
    save_visualization_analysis(visualization_analysis, filename=BASE_DIR / "src" / "EDA" / "Univariate_Analysis" / "Reports" / "visualization_analysis.json")

    # Render PNGs
    render_visualizations(df, output_dir=BASE_DIR / "src" / "EDA" / "Univariate_Analysis" / "Visuals", bins=30)