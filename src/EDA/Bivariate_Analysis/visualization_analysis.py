from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.EDA.Bivariate_Analysis.common import detect_column_type, generate_pairs, pair_id
from src. config import BASE_DIR, config


# ============================================================
# Visualization Specifications
# ============================================================

def numeric_numeric_visualizations(x: str, y: str) -> list[dict[str, Any]]:
    return [
        {"id": f"{x}_{y}_scatter", "type": "scatter", "purpose": "relationship", "x": x, "y": y},
        {"id": f"{x}_{y}_regression", "type": "regression", "purpose": "linear_relationship", "x": x, "y": y},
        {"id": f"{x}_{y}_hexbin", "type": "hexbin", "purpose": "density", "x": x, "y": y}
    ]


def numeric_categorical_visualizations(numeric: str, categorical: str) -> list[dict[str, Any]]:
    return [
        {"id": f"{numeric}_{categorical}_boxplot", "type": "boxplot", "purpose": "group_distribution", "x": categorical, "y": numeric},
        {"id": f"{numeric}_{categorical}_violin", "type": "violin", "purpose": "group_distribution", "x": categorical, "y": numeric},
        {"id": f"{numeric}_{categorical}_strip", "type": "strip", "purpose": "individual_observations", "x": categorical, "y": numeric}
    ]


def categorical_categorical_visualizations(x: str, y: str) -> list[dict[str, Any]]:
    return [
        {"id": f"{x}_{y}_grouped_bar", "type": "grouped_bar", "purpose": "category_relationship", "x": x, "y": y},
        {"id": f"{x}_{y}_stacked_bar", "type": "stacked_bar", "purpose": "category_composition", "x": x, "y": y},
        {"id": f"{x}_{y}_heatmap", "type": "heatmap", "purpose": "contingency_relationship", "x": x, "y": y}
    ]


def datetime_numeric_visualizations(datetime_col: str, numeric_col: str) -> list[dict[str, Any]]:
    return [
        {"id": f"{datetime_col}_{numeric_col}_timeline", "type": "line", "purpose": "time_trend", "x": datetime_col, "y": numeric_col},
        {"id": f"{datetime_col}_{numeric_col}_rolling", "type": "rolling_line", "purpose": "smoothed_time_trend", "x": datetime_col, "y": numeric_col}
    ]


def datetime_categorical_visualizations(datetime_col: str, categorical_col: str) -> list[dict[str, Any]]:
    return [
        {"id": f"{datetime_col}_{categorical_col}_count", "type": "time_count", "purpose": "category_frequency_over_time", "x": datetime_col, "hue": categorical_col},
        {"id": f"{datetime_col}_{categorical_col}_stacked", "type": "stacked_time_series", "purpose": "category_composition_over_time", "x": datetime_col, "hue": categorical_col}
    ]


def datetime_datetime_visualizations(x: str, y: str) -> list[dict[str, Any]]:
    return [
        {"id": f"{x}_{y}_gap", "type": "gap_distribution", "purpose": "time_difference", "x": x, "y": y}
    ]


# ============================================================
# Analysis Builder
# ============================================================

def analyze_visualizations(df: pd.DataFrame) -> dict[str, Any]:
    result = {"dataset": {"rows": int(len(df)), "columns": int(len(df.columns))}, "pairs": {}}

    for c1, c2 in generate_pairs(df):
        t1, t2 = detect_column_type(df[c1]), detect_column_type(df[c2])
        pair_key = (t1, t2)

        if pair_key == ("numeric", "numeric"):
            vis = numeric_numeric_visualizations(c1, c2)
        elif pair_key == ("numeric", "categorical"):
            vis = numeric_categorical_visualizations(c1, c2)
        elif pair_key == ("categorical", "numeric"):
            vis = numeric_categorical_visualizations(c2, c1)
        elif pair_key == ("categorical", "categorical"):
            vis = categorical_categorical_visualizations(c1, c2)
        elif pair_key == ("datetime", "numeric"):
            vis = datetime_numeric_visualizations(c1, c2)
        elif pair_key == ("numeric", "datetime"):
            vis = datetime_numeric_visualizations(c2, c1)
        elif pair_key == ("datetime", "categorical"):
            vis = datetime_categorical_visualizations(c1, c2)
        elif pair_key == ("categorical", "datetime"):
            vis = datetime_categorical_visualizations(c2, c1)
        elif pair_key == ("datetime", "datetime"):
            vis = datetime_datetime_visualizations(c1, c2)
        else:
            vis = []

        result["pairs"][pair_id(c1, c2)] = {
            "variables": {"column_1": c1, "column_2": c2, "type_1": t1, "type_2": t2},
            "visualizations": vis
        }

    return result


# ============================================================
# Plot Renderers
# ============================================================

def render_numeric_numeric(df: pd.DataFrame, x: str, y: str) -> plt.Figure:
    data = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(data[x], data[y], alpha=0.6)
    axes[0].set(title="Scatter Plot", xlabel=x, ylabel=y)

    sns.regplot(data=data, x=x, y=y, ax=axes[1], scatter_kws={"alpha": 0.5})
    axes[1].set_title("Regression Plot")

    axes[2].hexbin(data[x], data[y], gridsize=30, cmap="Blues")
    axes[2].set(title="Hexbin Density", xlabel=x, ylabel=y)

    plt.tight_layout()
    return fig


def render_numeric_categorical(df: pd.DataFrame, numeric: str, categorical: str, top_n: int = 20) -> plt.Figure:
    data = df[[numeric, categorical]].copy()
    data[numeric] = pd.to_numeric(data[numeric], errors="coerce")
    data = data.dropna()

    top_categories = data[categorical].value_counts().head(top_n).index
    data = data[data[categorical].isin(top_categories)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    sns.boxplot(data=data, x=categorical, y=numeric, ax=axes[0])
    axes[0].set_title("Boxplot")
    axes[0].tick_params(axis="x", rotation=45)

    sns.violinplot(data=data, x=categorical, y=numeric, ax=axes[1])
    axes[1].set_title("Violin Plot")
    axes[1].tick_params(axis="x", rotation=45)

    sns.stripplot(data=data, x=categorical, y=numeric, ax=axes[2], alpha=0.4, jitter=True)
    axes[2].set_title("Individual Observations")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def render_categorical_categorical(df: pd.DataFrame, x: str, y: str, top_n: int = 15) -> plt.Figure:
    data = df[[x, y]].dropna()
    top_x = data[x].value_counts().head(top_n).index
    top_y = data[y].value_counts().head(top_n).index
    data = data[data[x].isin(top_x) & data[y].isin(top_y)]

    table = pd.crosstab(data[x], data[y])
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    table.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Grouped Bar")
    axes[0].tick_params(axis="x", rotation=45)

    table.plot(kind="bar", stacked=True, ax=axes[1])
    axes[1].set_title("Stacked Bar")
    axes[1].tick_params(axis="x", rotation=45)

    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=axes[2])
    axes[2].set_title("Contingency Heatmap")

    plt.tight_layout()
    return fig


def render_datetime_numeric(df: pd.DataFrame, datetime_col: str, numeric_col: str) -> plt.Figure:
    data = pd.DataFrame({
        "date": pd.to_datetime(df[datetime_col], errors="coerce"),
        "value": pd.to_numeric(df[numeric_col], errors="coerce")
    }).dropna().sort_values("date")

    daily = data.set_index("date")["value"].resample("D").mean()
    rolling = daily.rolling(7, min_periods=1).mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].plot(daily.index, daily.values)
    axes[0].set(title="Time Trend", xlabel=datetime_col, ylabel=numeric_col)

    axes[1].plot(daily.index, daily.values, alpha=0.3, label="Daily")
    axes[1].plot(rolling.index, rolling.values, linewidth=2, label="7-day rolling mean")
    axes[1].set_title("Smoothed Time Trend")
    axes[1].legend()

    plt.tight_layout()
    return fig


def render_datetime_categorical(df: pd.DataFrame, datetime_col: str, categorical_col: str, top_n: int = 10) -> plt.Figure:
    data = pd.DataFrame({
        "date": pd.to_datetime(df[datetime_col], errors="coerce"),
        "category": df[categorical_col]
    }).dropna()

    top_categories = data["category"].value_counts().head(top_n).index
    data = data[data["category"].isin(top_categories)]

    monthly = pd.crosstab(data["date"].dt.to_period("M"), data["category"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    monthly.plot(ax=axes[0])
    axes[0].set(title="Category Frequency Over Time", xlabel="Month", ylabel="Count")

    monthly.plot(kind="area", stacked=True, ax=axes[1], alpha=0.75)
    axes[1].set(title="Category Composition Over Time", xlabel="Month", ylabel="Count")

    plt.tight_layout()
    return fig


def render_datetime_datetime(df: pd.DataFrame, x: str, y: str) -> plt.Figure:
    data = pd.DataFrame({
        "x": pd.to_datetime(df[x], errors="coerce"),
        "y": pd.to_datetime(df[y], errors="coerce")
    }).dropna()

    gap_hours = (data["y"] - data["x"]).dt.total_seconds() / 3600
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(gap_hours, bins=30, edgecolor="black", alpha=0.75)
    ax.set(title=f"Time Difference: {y} - {x}", xlabel="Difference (hours)", ylabel="Frequency")

    plt.tight_layout()
    return fig


# ============================================================
# Pair Rendering & Pipeline Orchestration
# ============================================================

def render_pair(df: pd.DataFrame, c1: str, c2: str) -> plt.Figure | None:
    t1, t2 = detect_column_type(df[c1]), detect_column_type(df[c2])
    pair_key = (t1, t2)

    if pair_key == ("numeric", "numeric"):
        return render_numeric_numeric(df, c1, c2)
    elif pair_key == ("numeric", "categorical"):
        return render_numeric_categorical(df, c1, c2)
    elif pair_key == ("categorical", "numeric"):
        return render_numeric_categorical(df, c2, c1)
    elif pair_key == ("categorical", "categorical"):
        return render_categorical_categorical(df, c1, c2)
    elif pair_key == ("datetime", "numeric"):
        return render_datetime_numeric(df, c1, c2)
    elif pair_key == ("numeric", "datetime"):
        return render_datetime_numeric(df, c2, c1)
    elif pair_key == ("datetime", "categorical"):
        return render_datetime_categorical(df, c1, c2)
    elif pair_key == ("categorical", "datetime"):
        return render_datetime_categorical(df, c2, c1)
    elif pair_key == ("datetime", "datetime"):
        return render_datetime_datetime(df, c1, c2)
    
    return None


def render_visualizations(df: pd.DataFrame, output_dir: Path = BASE_DIR / "src" / "EDA" / "Bivariate_Analysis" / "Visuals") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for c1, c2 in generate_pairs(df):
        fig = render_pair(df, c1, c2)
        if fig is not None:
            fig.savefig(output / f"{c1}__{c2}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def save_visualization_analysis(analysis: dict[str, Any], filename: Path = BASE_DIR / "src" / "EDA" / "Bivariate_Analysis" / "Reports" / "bivariate_visualization.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    analysis = analyze_visualizations(df)
    save_visualization_analysis(analysis, filename=BASE_DIR / "src" / "EDA" / "Bivariate_Analysis" / "Reports" / "bivariate_visualization.json")
    render_visualizations(df, output_dir=BASE_DIR / "src" / "EDA" / "Bivariate_Analysis" / "Visuals")

    print(json.dumps(analysis, indent=2, ensure_ascii=False))