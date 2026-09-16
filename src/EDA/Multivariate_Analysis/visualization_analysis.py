from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.EDA.Multivariate_Analysis.common import dataset_metadata, get_column_groups
from src.config import BASE_DIR


# ============================================================
# Visualization Specifications
# ============================================================

def numeric_multivariate_specs(columns: list[str]) -> list[dict[str, Any]]:
    if len(columns) < 2:
        return []
    return [
        {
            "id": "numeric_correlation_pearson",
            "type": "correlation_heatmap",
            "purpose": "numeric_relationships",
            "method": "pearson",
            "variables": columns
        },
        {
            "id": "numeric_correlation_spearman",
            "type": "correlation_heatmap",
            "purpose": "rank_relationships",
            "method": "spearman",
            "variables": columns
        },
        {
            "id": "numeric_pairplot",
            "type": "pairplot",
            "purpose": "multivariate_numeric_relationships",
            "variables": columns
        },
        {
            "id": "numeric_covariance",
            "type": "covariance_heatmap",
            "purpose": "covariance_structure",
            "variables": columns
        },
        {
            "id": "numeric_pca",
            "type": "pca_projection",
            "purpose": "dimensionality_reduction",
            "variables": columns
        }
    ]


def categorical_multivariate_specs(columns: list[str]) -> list[dict[str, Any]]:
    if len(columns) < 2:
        return []
    return [
        {
            "id": "categorical_association",
            "type": "association_heatmap",
            "purpose": "categorical_relationships",
            "variables": columns,
            "method": "cramers_v"
        },
        {
            "id": "categorical_frequency",
            "type": "frequency_heatmap",
            "purpose": "category_combinations",
            "variables": columns
        }
    ]


def grouped_numeric_specs(numeric_columns: list[str], categorical_columns: list[str]) -> list[dict[str, Any]]:
    specs = []
    for numeric in numeric_columns:
        for categorical in categorical_columns:
            specs.append({
                "id": f"{numeric}_by_{categorical}",
                "type": "grouped_distribution",
                "purpose": "numeric_distribution_by_category",
                "numeric": numeric,
                "category": categorical
            })
    return specs


def temporal_specs(datetime_columns: list[str], numeric_columns: list[str]) -> list[dict[str, Any]]:
    specs = []
    for datetime_column in datetime_columns:
        if len(numeric_columns) == 0:
            continue
        specs.append({
            "id": f"{datetime_column}_numeric_trends",
            "type": "multivariate_time_series",
            "purpose": "multiple_numeric_variables_over_time",
            "datetime": datetime_column,
            "variables": numeric_columns
        })
    return specs


# ============================================================
# Analysis Builder
# ============================================================

def analyze_visualizations(df: pd.DataFrame) -> dict[str, Any]:
    groups = get_column_groups(df)
    numeric_columns = groups["numeric"]
    categorical_columns = groups["categorical"]
    datetime_columns = groups["datetime"]

    result = {
        "analysis_type": "multivariate",
        "dataset": dataset_metadata(df),
        "visualizations": []
    }

    result["visualizations"].extend(numeric_multivariate_specs(numeric_columns))
    result["visualizations"].extend(categorical_multivariate_specs(categorical_columns))
    result["visualizations"].extend(grouped_numeric_specs(numeric_columns, categorical_columns))
    result["visualizations"].extend(temporal_specs(datetime_columns, numeric_columns))
    return result


# ============================================================
# Plot Renderers
# ============================================================

def render_correlation_heatmap(df: pd.DataFrame, method: str = "pearson") -> plt.Figure | None:
    groups = get_column_groups(df)
    columns = groups["numeric"]
    if len(columns) < 2:
        return None

    data = df[columns].apply(pd.to_numeric, errors="coerce")
    correlation = data.corr(method=method)

    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.7), max(6, len(columns) * 0.6)))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        ax=ax
    )
    ax.set_title(f"{method.title()} Correlation Matrix")
    plt.tight_layout()
    return fig


def render_covariance_heatmap(df: pd.DataFrame) -> plt.Figure | None:
    columns = get_column_groups(df)["numeric"]
    if len(columns) < 2:
        return None

    data = df[columns].apply(pd.to_numeric, errors="coerce")
    covariance = data.cov()

    fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.7), max(6, len(columns) * 0.6)))
    sns.heatmap(covariance, annot=True, fmt=".2f", cmap="viridis", ax=ax)
    ax.set_title("Covariance Matrix")
    plt.tight_layout()
    return fig


def render_pairplot(df: pd.DataFrame, max_variables: int = 8) -> plt.Figure | None:
    columns = get_column_groups(df)["numeric"]
    if len(columns) < 2:
        return None

    columns = columns[:max_variables]
    data = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) == 0:
        return None

    grid = sns.pairplot(data)
    grid.fig.suptitle("Multivariate Numeric Pairplot", y=1.02)
    return grid.fig


def render_pca(df: pd.DataFrame) -> plt.Figure | None:
    columns = get_column_groups(df)["numeric"]
    if len(columns) < 2:
        return None

    data = df[columns].apply(pd.to_numeric, errors="coerce").dropna()
    if len(data) < 3:
        return None

    X = data.values
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=1)
    std[std == 0] = 1
    X = (X - mean) / std

    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    components = Vt[:2]
    projection = X @ components.T

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(projection[:, 0], projection[:, 1], alpha=0.6)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set(title="PCA Projection", xlabel="PC1", ylabel="PC2")
    plt.tight_layout()
    return fig


def render_categorical_heatmap(df: pd.DataFrame, max_categories: int = 10) -> plt.Figure | None:
    columns = get_column_groups(df)["categorical"]
    if len(columns) < 2:
        return None

    x, y = columns[:2]
    data = df[[x, y]].dropna()
    top_x = data[x].value_counts().head(max_categories).index
    top_y = data[y].value_counts().head(max_categories).index
    data = data[data[x].isin(top_x) & data[y].isin(top_y)]

    table = pd.crosstab(data[x], data[y])
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(table, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(f"Category Combination Frequency: {x} × {y}")
    plt.tight_layout()
    return fig


def render_grouped_distribution(df: pd.DataFrame, numeric: str, categorical: str, max_categories: int = 15) -> plt.Figure | None:
    data = df[[numeric, categorical]].copy()
    data[numeric] = pd.to_numeric(data[numeric], errors="coerce")
    data = data.dropna()

    top_categories = data[categorical].value_counts().head(max_categories).index
    data = data[data[categorical].isin(top_categories)]
    if data.empty:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.boxplot(data=data, x=categorical, y=numeric, ax=axes[0])
    axes[0].set_title(f"{numeric} by {categorical}")
    axes[0].tick_params(axis="x", rotation=45)

    sns.violinplot(data=data, x=categorical, y=numeric, ax=axes[1])
    axes[1].set_title(f"Distribution of {numeric} by {categorical}")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def render_temporal_multivariate(df: pd.DataFrame, datetime_column: str) -> plt.Figure | None:
    numeric_columns = get_column_groups(df)["numeric"]
    if len(numeric_columns) == 0:
        return None

    data = df.copy()
    data[datetime_column] = pd.to_datetime(data[datetime_column], errors="coerce")
    data = data.dropna(subset=[datetime_column]).set_index(datetime_column)
    if data.empty:
        return None

    numeric = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    monthly = numeric.resample("ME").mean()
    if monthly.empty:
        return None

    normalized = (monthly - monthly.mean()) / monthly.std()

    fig, ax = plt.subplots(figsize=(14, 7))
    for column in normalized.columns:
        ax.plot(normalized.index, normalized[column], label=column)

    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set(
        title="Normalized Multivariate Time Series",
        xlabel=datetime_column,
        ylabel="Standardized value"
    )
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig


# ============================================================
# Pipeline Orchestration
# ============================================================

def render_visualizations(df: pd.DataFrame, output_dir: Path = BASE_DIR / "src" / "EDA" / "Multivariate_Analysis" / "Visuals") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    groups = get_column_groups(df)

    # 1. Pearson correlation
    fig = render_correlation_heatmap(df, method="pearson")
    if fig is not None:
        fig.savefig(output / "pearson_correlation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 2. Spearman correlation
    fig = render_correlation_heatmap(df, method="spearman")
    if fig is not None:
        fig.savefig(output / "spearman_correlation.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Covariance
    fig = render_covariance_heatmap(df)
    if fig is not None:
        fig.savefig(output / "covariance.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 4. Pairplot
    fig = render_pairplot(df)
    if fig is not None:
        fig.savefig(output / "pairplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 5. PCA
    fig = render_pca(df)
    if fig is not None:
        fig.savefig(output / "pca.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 6. Categorical heatmap
    fig = render_categorical_heatmap(df)
    if fig is not None:
        fig.savefig(output / "categorical_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 7. Grouped distributions
    for num_col in groups["numeric"]:
        for cat_col in groups["categorical"]:
            fig = render_grouped_distribution(df, num_col, cat_col)
            if fig is not None:
                fig.savefig(output / f"{num_col}_by_{cat_col}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)

    # 8. Temporal multivariate time series
    for dt_col in groups["datetime"]:
        fig = render_temporal_multivariate(df, dt_col)
        if fig is not None:
            fig.savefig(output / f"{dt_col}_temporal_multivariate.png", dpi=150, bbox_inches="tight")
            plt.close(fig)


def save_visualization_analysis(analysis: dict[str, Any], filename: Path = BASE_DIR / "src" / "EDA" / "Multivariate_Analysis" / "Reports" / "multivariate_visualization.json") -> None:
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data.csv")

    analysis = analyze_visualizations(df)
    save_visualization_analysis(analysis, filename=BASE_DIR / "src" / "EDA" / "Multivariate_Analysis" / "Reports" / "multivariate_visualization.json")
    render_visualizations(df, output_dir=BASE_DIR / "src" / "EDA" / "Multivariate_Analysis" / "Visuals")

    print(json.dumps(analysis, indent=2, ensure_ascii=False))