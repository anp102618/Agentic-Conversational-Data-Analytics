from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import seaborn as sns
from scipy.stats import pearsonr, spearmanr, chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import clone
from sklearn.inspection import permutation_importance as sklearn_permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import (KFold,StratifiedKFold,cross_validate)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,mean_squared_error, precision_score, r2_score,recall_score, roc_auc_score)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.EDA.Target_Feature_Analysis.common import (dataset_target_metadata, get_feature_types, infer_target_type, make_json_safe)
from src.config import BASE_DIR, config


class TargetEDA:
    """
    Main target-analysis engine.

    Each analytical component will be implemented as an
    independent method.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target: str,
        output_dir: str | Path = BASE_DIR / "src" / "EDA" / "Target_Feature_Analysis" / "Reports",
        classification_threshold: int = 10,
        random_state: int = 42,
    ) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("Input dataframe is empty.")
        if not isinstance(target, str):
            raise TypeError("target must be a column name string.")
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' was not found in the dataframe.")

        # Core Inputs & Config
        self.df = df.copy()
        self.target = target
        self.y = self.df[target]
        self.X = self.df.drop(columns=[target])
        self.classification_threshold = classification_threshold
        self.random_state = random_state

        # Output Directories
        self.output_dir = Path(output_dir)
        self.plots_dir = BASE_DIR / "src" / "EDA" / "Target_Feature_Analysis" / "Visuals"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        # Target & Feature Typing
        self.target_type = infer_target_type(self.y, classification_threshold=self.classification_threshold)
        
        feature_types = get_feature_types(self.X)
        self.numeric_features = feature_types["numeric"]
        self.categorical_features = feature_types["categorical"]
        self.datetime_features = feature_types["datetime"]

        # Metadata & Containers
        self.metadata = dataset_target_metadata(df=self.df, target=self.target, target_type=self.target_type)
        self.results: dict[str, Any] = {}
        self.component_status: dict[str, dict[str, Any]] = {}

    def info(self) -> dict[str, Any]:
        """Return basic information about the analysis object without running analytics."""
        return {
            "target": self.target,
            "target_type": self.target_type,
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "feature_count": len(self.X.columns),
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "datetime_features": self.datetime_features,
            "output_dir": str(self.output_dir),
            "plots_dir": str(self.plots_dir),
        }


    def target_distribution(
        self,
        filename: str = "target_distribution.png",
        bins: int = 30,
        figsize: tuple[int, int] = (10, 6),
    ) -> dict[str, Any]:
        """Analyze and visualize the target distribution for classification or regression."""
        

        component_name = "target_distribution"
        plot_path = self.plots_dir / filename

        try:
            y = self.y.dropna()
            if y.empty:
                raise ValueError("Target contains no non-missing observations.")

            if self.target_type == "classification":
                counts = y.value_counts(dropna=False)
                percentages = y.value_counts(normalize=True, dropna=False) * 100

                class_counts = {str(k): int(v) for k, v in counts.items()}
                class_percentages = {str(k): round(float(v), 4) for k, v in percentages.items()}
                
                n_classes = int(y.nunique(dropna=True))
                majority_class, majority_count = str(counts.index[0]), int(counts.iloc[0])
                minority_class, minority_count = str(counts.index[-1]), int(counts.iloc[-1])
                imbalance_ratio = round(majority_count / minority_count, 4) if minority_count > 0 else None

                result = {
                    "component": component_name,
                    "status": "success",
                    "target": self.target,
                    "target_type": self.target_type,
                    "count": int(len(y)),
                    "number_of_classes": n_classes,
                    "class_counts": class_counts,
                    "class_percentages": class_percentages,
                    "majority_class": majority_class,
                    "majority_count": majority_count,
                    "minority_class": minority_class,
                    "minority_count": minority_count,
                    "imbalance_ratio": imbalance_ratio,
                    "plot": str(plot_path),
                }

                fig, ax = plt.subplots(figsize=figsize)
                bars = ax.bar(class_counts.keys(), class_counts.values())
                ax.set_title(f"Target Distribution: {self.target}")
                ax.set_xlabel("Target Class")
                ax.set_ylabel("Count")

                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height,
                        f"{int(height):,}",
                        ha="center",
                        va="bottom",
                    )

                plt.tight_layout()
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            elif self.target_type == "regression":
                y_numeric = pd.to_numeric(y, errors="coerce").dropna()
                if y_numeric.empty:
                    raise ValueError("Regression target could not be converted to numeric values.")

                q1 = float(y_numeric.quantile(0.25))
                q3 = float(y_numeric.quantile(0.75))
                mean_val = float(y_numeric.mean())
                median_val = float(y_numeric.median())

                result = {
                    "component": component_name,
                    "status": "success",
                    "target": self.target,
                    "target_type": self.target_type,
                    "count": int(len(y_numeric)),
                    "mean": mean_val,
                    "median": median_val,
                    "std": float(y_numeric.std()),
                    "min": float(y_numeric.min()),
                    "max": float(y_numeric.max()),
                    "q1": q1,
                    "q3": q3,
                    "iqr": float(q3 - q1),
                    "skewness": float(y_numeric.skew()),
                    "kurtosis": float(y_numeric.kurtosis()),
                    "plot": str(plot_path),
                }

                fig, axes = plt.subplots(1, 2, figsize=figsize)

                # Histogram
                axes[0].hist(y_numeric, bins=bins, edgecolor="black", alpha=0.75)
                axes[0].axvline(mean_val, linestyle="--", linewidth=2, label=f"Mean: {mean_val:.3f}")
                axes[0].axvline(median_val, linestyle=":", linewidth=2, label=f"Median: {median_val:.3f}")
                axes[0].set_title("Target Histogram")
                axes[0].set_xlabel(self.target)
                axes[0].set_ylabel("Frequency")
                axes[0].legend()

                # Boxplot
                axes[1].boxplot(y_numeric, vert=True)
                axes[1].set_title("Target Boxplot")
                axes[1].set_ylabel(self.target)

                plt.suptitle(f"Target Distribution: {self.target}")
                plt.tight_layout()
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")

            self.results[component_name] = result
            self.component_status[component_name] = {"status": "success", "plot": str(plot_path)}
            return result

        except Exception as exc:
            error_result = {
                "component": component_name,
                "status": "error",
                "target": self.target,
                "target_type": self.target_type,
                "error": str(exc),
            }
            self.results[component_name] = error_result
            self.component_status[component_name] = {"status": "error", "error": str(exc)}
            return error_result


    def missingness(
        self,
        filename: str = "missingness.png",
        figsize: tuple[int, int] = (12, 7),
        top_n: int | None = 30,
    ) -> dict[str, Any]:
        """Analyze missing values in the dataset and save a bar chart visualization."""
        

        component_name = "missingness"
        df = self.df

        try:
            missing_counts = df.isna().sum()
            missing_percentages = df.isna().mean() * 100
            missing_columns = missing_counts[missing_counts > 0].sort_values(ascending=False)

            feature_missing_counts = self.X.isna().sum()
            target_missing_count = int(self.y.isna().sum())
            target_missing_percentage = float(self.y.isna().mean() * 100)

            total_rows = len(df)
            total_cols = len(df.columns)
            total_cells = int(total_rows * total_cols)
            total_missing_cells = int(missing_counts.sum())
            
            rows_with_missing = int(df.isna().any(axis=1).sum())
            complete_rows = int(df.notna().all(axis=1).sum())
            
            rows_with_missing_percentage = float(rows_with_missing / total_rows * 100) if total_rows > 0 else 0.0
            overall_missing_percentage = float(total_missing_cells / total_cells * 100) if total_cells > 0 else 0.0

            # Column-level details via dictionary comprehension
            column_details = {
                str(col): {
                    "missing_count": int(missing_counts[col]),
                    "missing_percentage": round(float(missing_percentages[col]), 4),
                    "non_missing_count": int(df[col].notna().sum()),
                }
                for col in df.columns
            }

            sorted_missing = missing_percentages[missing_percentages > 0].sort_values(ascending=False)

            # Visualization
            plot_path = self.plots_dir / filename
            fig, ax = plt.subplots(figsize=figsize)

            if sorted_missing.empty:
                ax.text(0.5, 0.5, "No missing values found", ha="center", va="center", fontsize=16, transform=ax.transAxes)
                ax.set_title("Missingness Analysis")
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                if top_n is not None:
                    if top_n <= 0:
                        raise ValueError("top_n must be greater than 0 or None.")
                    plot_missing = sorted_missing.head(top_n)
                else:
                    plot_missing = sorted_missing

                bars = ax.bar(plot_missing.index.astype(str), plot_missing.values)
                ax.set_title("Missing Values by Column")
                ax.set_xlabel("Column")
                ax.set_ylabel("Missing Percentage (%)")
                ax.tick_params(axis="x", rotation=75)

                for bar, value in zip(bars, plot_missing.values):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{value:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                ax.grid(axis="y", alpha=0.25)

            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Result construction
            result = {
                "component": component_name,
                "status": "success",
                "total_rows": total_rows,
                "total_columns": total_cols,
                "total_cells": total_cells,
                "total_missing_cells": total_missing_cells,
                "overall_missing_percentage": round(overall_missing_percentage, 4),
                "rows_with_missing": rows_with_missing,
                "rows_with_missing_percentage": round(rows_with_missing_percentage, 4),
                "complete_rows": complete_rows,
                "target_missing_count": target_missing_count,
                "target_missing_percentage": round(target_missing_percentage, 4),
                "feature_missing_count": int(feature_missing_counts.sum()),
                "columns_with_missing": int(len(missing_columns)),
                "missing_columns": [str(c) for c in missing_columns.index],
                "column_details": column_details,
                "plot": str(plot_path),
            }

            self.results[component_name] = result
            self.component_status[component_name] = {"status": "success", "plot": str(plot_path)}
            return result

        except Exception as exc:
            error_result = {
                "component": component_name,
                "status": "error",
                "error": str(exc),
            }
            self.results[component_name] = error_result
            self.component_status[component_name] = {"status": "error", "error": str(exc)}
            return error_result


    def correlation_analysis(
        self,
        filename: str = "correlation.png",
        figsize: tuple[int, int] = (12, 8),
        top_n: int = 20,
    ) -> dict[str, Any]:
        """
        Analyze relationships between numeric features and
        the target using Pearson and Spearman correlation.
        """
        

        comp = "correlation"
        try:
            if top_n <= 0:
                raise ValueError("top_n must be greater than 0.")
            if not self.numeric_features:
                raise ValueError("No numeric features available for correlation analysis.")

            y = self.y.copy()
            if self.target_type == "regression":
                y_numeric = pd.to_numeric(y, errors="coerce")
                target_encoding = "original_numeric"
            else:
                y_numeric, uniques = pd.factorize(y, sort=True)
                y_numeric = pd.Series(y_numeric, index=y.index, dtype=float).replace(-1, np.nan)
                target_encoding = {"method": "factorize", "classes": [str(v) for v in uniques]}

            pearson_results, spearman_results = {}, {}

            for feature in self.numeric_features:
                if feature not in self.X.columns:
                    continue
                x = pd.to_numeric(self.X[feature], errors="coerce")
                valid_mask = x.notna() & y_numeric.notna()
                x_valid, y_valid = x.loc[valid_mask], y_numeric.loc[valid_mask]
                n_val = int(len(x_valid))

                if n_val < 3 or x_valid.nunique() < 2 or y_valid.nunique() < 2:
                    status = "insufficient_data" if n_val < 3 else ("constant_feature" if x_valid.nunique() < 2 else "constant_target")
                    res_stub = {"correlation": None, "p_value": None, "n": n_val, "status": status}
                    pearson_results[str(feature)] = res_stub
                    spearman_results[str(feature)] = res_stub
                    continue

                try:
                    p_corr, p_val = pearsonr(x_valid, y_valid)
                    pearson_results[str(feature)] = {
                        "correlation": float(p_corr), "p_value": float(p_val),
                        "absolute_correlation": float(abs(p_corr)), "n": n_val, "status": "success"
                    }
                except Exception as exc:
                    pearson_results[str(feature)] = {"correlation": None, "p_value": None, "n": n_val, "status": "error", "error": str(exc)}

                try:
                    s_corr, s_val = spearmanr(x_valid, y_valid)
                    spearman_results[str(feature)] = {
                        "correlation": float(s_corr), "p_value": float(s_val),
                        "absolute_correlation": float(abs(s_corr)), "n": n_val, "status": "success"
                    }
                except Exception as exc:
                    spearman_results[str(feature)] = {"correlation": None, "p_value": None, "n": n_val, "status": "error", "error": str(exc)}

            def sorted_correlations(corr_dict):
                valid = [
                    {"feature": f, **vals} for f, vals in corr_dict.items()
                    if vals.get("correlation") is not None
                ]
                valid.sort(key=lambda i: abs(i["correlation"]), reverse=True)
                return valid

            pearson_sorted = sorted_correlations(pearson_results)
            spearman_sorted = sorted_correlations(spearman_results)
            strongest_pearson = pearson_sorted[:top_n]
            strongest_spearman = spearman_sorted[:top_n]

            vis_features = list(set([i["feature"] for i in strongest_pearson] + [i["feature"] for i in strongest_spearman]))
            vis_features.sort(
                key=lambda f: max(
                    abs(pearson_results[f]["correlation"]) if pearson_results[f].get("correlation") is not None else 0,
                    abs(spearman_results[f]["correlation"]) if spearman_results[f].get("correlation") is not None else 0
                ),
                reverse=True
            )

            plot_path = self.plots_dir / filename
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            if vis_features:
                plot_data = pd.DataFrame({
                    "Pearson": [pearson_results[f].get("correlation") for f in vis_features],
                    "Spearman": [spearman_results[f].get("correlation") for f in vis_features],
                }, index=vis_features)

                sns.heatmap(plot_data, annot=True, fmt=".3f", cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=axes[0], linewidths=0.5)
                axes[0].set(title="Pearson vs Spearman", xlabel="Method", ylabel="Feature")

                plot_data.abs().sort_values("Pearson", ascending=True).plot(kind="barh", ax=axes[1])
                axes[1].set(title="Absolute Correlation", xlabel="|Correlation|", ylabel="Feature", xlim=(0, 1))
                axes[1].grid(axis="x", alpha=0.25)
            else:
                for ax in axes:
                    ax.text(0.5, 0.5, "No valid correlations", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([])
                    ax.set_yticks([])

            plt.suptitle(f"Target Correlation Analysis: {self.target}")
            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            result = {
                "component": comp, "status": "success", "target": self.target,
                "target_type": self.target_type, "target_encoding": target_encoding,
                "numeric_feature_count": int(len(self.numeric_features)),
                "pearson": pearson_results, "spearman": spearman_results,
                "strongest_pearson": strongest_pearson, "strongest_spearman": strongest_spearman,
                "plot": str(plot_path),
            }
            self.results[comp] = result
            self.component_status[comp] = {"status": "success", "plot": str(plot_path)}
            return result

        except Exception as exc:
            err_res = {"component": comp, "status": "error", "error": str(exc)}
            self.results[comp] = err_res
            self.component_status[comp] = {"status": "error", "error": str(exc)}
            return err_res


    def categorical_analysis(
        self,
        output_subdir: str = "categorical",
        top_n: int = 20,
        min_count: int = 1,
        figsize: tuple[int, int] = (10, 6),
    ) -> dict[str, Any]:
        """Analyze categorical features against the target."""
     

        comp_name = "categorical_analysis"
        
        if top_n <= 0 or min_count <= 0:
            raise ValueError("top_n and min_count must be greater than 0.")

        cat_features = [f for f in self.categorical_features if f in self.X.columns]
        if not cat_features:
            res = {"component": comp_name, "status": "skipped", "reason": "No categorical features are available.", "features": {}}
            self.results[comp_name] = res
            self.component_status[comp_name] = {"status": "skipped", "reason": res["reason"]}
            return res

        plot_dir = self.plots_dir / output_subdir
        plot_dir.mkdir(parents=True, exist_ok=True)
        feature_results = {}

        for feature in cat_features:
            try:
                series = self.X[feature].astype("object").fillna("__MISSING__").astype(str)
                counts = series.value_counts(dropna=False)
                counts = counts[counts >= min_count].head(top_n)
                selected = counts.index.tolist()
                
                feature_series = series.loc[series.isin(selected)]
                plot_path = plot_dir / f"{feature}_categorical.png"
                fig, ax = plt.subplots(figsize=figsize)

                if self.target_type == "classification":
                    target_series = self.y.loc[feature_series.index]
                    valid = target_series.notna()
                    f_val, t_val = feature_series.loc[valid], target_series.loc[valid]
                    
                    if f_val.empty or t_val.empty:
                        raise ValueError("No valid observations remain after removing missing target values.")

                    contingency = pd.crosstab(f_val, t_val)
                    row_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                    
                    # Chi-square & Cramér's V
                    chi2, p_val, dof, cramers_v = None, None, None, None
                    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
                        try:
                            chi2, p_val, dof, _ = chi2_contingency(contingency)
                            n = contingency.values.sum()
                            denom = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
                            if n > 0 and denom > 0:
                                cramers_v = float(np.sqrt((chi2 / n) / denom))
                        except Exception:
                            pass

                    # Plotting
                    row_pct.plot(kind="bar", stacked=True, ax=ax)
                    ax.set_title(f"{feature} vs {self.target}")
                    ax.set_ylabel("Target Distribution (%)")
                    
                    feature_results[str(feature)] = {
                        "status": "success", "feature": str(feature), "target_type": "classification",
                        "n_observations": int(len(f_val)), "n_categories": int(len(selected)),
                        "categories": [str(c) for c in selected],
                        "category_counts": {str(k): int(v) for k, v in f_val.value_counts().items()},
                        "class_counts": {str(cat): {str(k): int(v) for k, v in contingency.loc[cat].items()} for cat in contingency.index},
                        "class_percentages": {str(cat): {str(k): round(float(v), 4) for k, v in row_pct.loc[cat].items()} for cat in row_pct.index},
                        "chi_square": float(chi2) if chi2 is not None else None,
                        "chi_square_p_value": float(p_val) if p_val is not None else None,
                        "degrees_of_freedom": int(dof) if dof is not None else None,
                        "cramers_v": float(cramers_v) if cramers_v is not None else None,
                        "plot": str(plot_path),
                    }
                else:
                    target_numeric = pd.to_numeric(self.y, errors="coerce")
                    valid = feature_series.index.isin(target_numeric.dropna().index)
                    f_val, t_val = feature_series.loc[valid], target_numeric.loc[valid.index if hasattr(valid, 'index') else feature_series.index]
                    # Fallback alignment if needed
                    t_val = target_numeric.loc[f_val.index]

                    if f_val.empty or t_val.empty:
                        raise ValueError("No valid observations remain for regression analysis.")

                    plot_df = pd.DataFrame({"category": f_val, "target": t_val})
                    sns.boxplot(data=plot_df, x="category", y="target", ax=ax)
                    ax.set_title(f"{self.target} by {feature}")
                    ax.set_ylabel(self.target)

                    grouped = plot_df.groupby("category")["target"]
                    group_stats = {
                        str(cat): {
                            "count": int(vals.count()), "mean": float(vals.mean()),
                            "median": float(vals.median()), "std": float(vals.std()) if len(vals) > 1 else None,
                            "min": float(vals.min()), "max": float(vals.max())
                        } for cat, vals in grouped
                    }

                    feature_results[str(feature)] = {
                        "status": "success", "feature": str(feature), "target_type": "regression",
                        "n_observations": int(len(f_val)), "n_categories": int(len(group_stats)),
                        "categories": [str(c) for c in group_stats],
                        "group_statistics": group_stats,
                        "plot": str(plot_path),
                    }

                # Common plot formatting & saving
                ax.set_xlabel(feature)
                ax.tick_params(axis="x", rotation=60)
                ax.grid(axis="y", alpha=0.2)
                if self.target_type == "classification":
                    ax.legend(title=self.target, bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

            except Exception as exc:
                feature_results[str(feature)] = {"status": "error", "feature": str(feature), "error": str(exc)}

        final_result = {
            "component": comp_name, "status": "success",
            "features": feature_results
        }
        self.results[comp_name] = final_result
        self.component_status[comp_name] = {"status": "success"}
        return final_result



    def preprocessing(self, fit: bool = True) -> dict[str, Any]:
        """Build and optionally fit the preprocessing pipeline."""
        

        comp_name = "preprocessing"

        try:
            numeric_features = [f for f in self.numeric_features if f in self.X.columns]
            categorical_features = [f for f in self.categorical_features if f in self.X.columns]

            if not numeric_features and not categorical_features:
                res = {
                    "component": comp_name,
                    "status": "skipped",
                    "reason": "No numeric or categorical features are available.",
                }
                self.results[comp_name] = res
                self.component_status[comp_name] = {"status": "skipped", "reason": res["reason"]}
                return res

            if fit:
                transformers = []
                if numeric_features:
                    transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
                if categorical_features:
                    transformers.append(("categorical", Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                    ]), categorical_features))

                self.preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
                X_transformed = self.preprocessor.fit_transform(self.X)
            else:
                if not hasattr(self, "preprocessor"):
                    raise RuntimeError("No fitted preprocessor exists. Run preprocessing(fit=True) first.")
                X_transformed = self.preprocessor.transform(self.X)

            X_transformed = np.asarray(X_transformed)
            self.X_transformed = X_transformed

            # Feature names extraction
            try:
                transformed_names = self.preprocessor.get_feature_names_out().tolist()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]
            
            self.transformed_feature_names = transformed_names

            # Clean feature names & original mapping
            cleaned_names = []
            original_feature_mapping = {}
            sorted_columns = sorted(self.X.columns, key=lambda v: len(str(v)), reverse=True)

            for name in transformed_names:
                clean = name.replace("numeric__", "").replace("categorical__", "")
                cleaned_names.append(clean)
                
                orig_feat = clean
                for col in sorted_columns:
                    col_str = str(col)
                    if clean == col_str or clean.startswith(f"{col_str}_") or clean.startswith(f"{col_str}=") or clean.startswith(f"{col_str}["):
                        orig_feat = col_str
                        break
                original_feature_mapping[name] = orig_feat

            self.cleaned_feature_names = cleaned_names
            self.original_feature_mapping = original_feature_mapping

            is_sparse = hasattr(X_transformed, "toarray") if not isinstance(X_transformed, np.ndarray) else False

            result = {
                "component": comp_name,
                "status": "success",
                "fit": bool(fit),
                "original_feature_count": int(len(self.X.columns)),
                "numeric_feature_count": int(len(numeric_features)),
                "categorical_feature_count": int(len(categorical_features)),
                "transformed_feature_count": int(X_transformed.shape[1]),
                "transformed_shape": [int(v) for v in X_transformed.shape],
                "numeric_features": [str(f) for f in numeric_features],
                "categorical_features": [str(f) for f in categorical_features],
                "numeric_imputation": "median",
                "categorical_imputation": "most_frequent",
                "categorical_encoding": "one_hot",
                "handle_unknown": "ignore",
                "transformed_feature_names": [str(n) for n in transformed_names],
                "cleaned_feature_names": [str(n) for n in cleaned_names],
                "original_feature_mapping": original_feature_mapping,
                "sparse_output": is_sparse,
            }

            self.results[comp_name] = result
            self.component_status[comp_name] = {
                "status": "success",
                "transformed_feature_count": int(X_transformed.shape[1]),
            }
            return result

        except Exception as exc:
            error_result = {
                "component": comp_name,
                "status": "error",
                "error": str(exc),
            }
            self.results[comp_name] = error_result
            self.component_status[comp_name] = {
                "status": "error",
                "error": str(exc),
            }
            return error_result


    # ========================================================
    # COMPONENT 6: RANDOM FOREST
    # ========================================================

    def random_forest(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str | int | float | None = "sqrt",
        class_weight: str | dict | None = "balanced",
        top_n: int = 20,
        filename: str = "random_forest_importance.png",
        figsize: tuple[int, int] = (12, 8),
    ) -> dict[str, Any]:
        """Train a Random Forest model and calculate feature importance."""
        

        comp_name = "random_forest"

        try:
            # --- Validation ---
            if n_estimators <= 0:
                raise ValueError("n_estimators must be greater than 0.")
            if min_samples_split < 2:
                raise ValueError("min_samples_split must be at least 2.")
            if min_samples_leaf < 1:
                raise ValueError("min_samples_leaf must be at least 1.")
            if top_n <= 0:
                raise ValueError("top_n must be greater than 0.")

            # --- Data Preparation ---
            X = self.X.copy()
            valid_mask = self.y.notna()
            X, y = X.loc[valid_mask], self.y.loc[valid_mask]

            if len(X) < 2:
                raise ValueError("Not enough observations with a non-missing target.")

            # --- Target Model Assignment ---
            if self.target_type == "classification":
                y_model = y.copy()
                model = RandomForestClassifier(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    max_features=max_features, class_weight=class_weight,
                    random_state=self.random_state, n_jobs=-1
                )
            elif self.target_type == "regression":
                y_model = pd.to_numeric(y, errors="coerce")
                num_mask = y_model.notna()
                X, y_model = X.loc[num_mask], y_model.loc[num_mask]
                
                if len(X) < 2:
                    raise ValueError("Regression target does not contain enough valid numeric observations.")
                
                model = RandomForestRegressor(
                    n_estimators=n_estimators, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    max_features=max_features, random_state=self.random_state, n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")

            # --- Feature Pipelines ---
            num_feats = [f for f in self.numeric_features if f in X.columns]
            cat_feats = [f for f in self.categorical_features if f in X.columns]
            transformers = []

            if num_feats:
                transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_feats))
            if cat_feats:
                cat_pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ])
                transformers.append(("categorical", cat_pipe, cat_feats))

            if not transformers:
                raise ValueError("No numeric or categorical features are available for Random Forest.")

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            # --- Fit & Extract ---
            pipeline.fit(X, y_model)
            self.random_forest_model = pipeline
            fit_prep, fit_model = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]

            try:
                transformed_names = fit_prep.get_feature_names_out().tolist()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(len(fit_model.feature_importances_))]

            importances = fit_model.feature_importances_
            if len(importances) != len(transformed_names):
                raise RuntimeError("Number of feature importances does not match transformed feature names.")

            # --- Transformed & Original Feature Processing ---
            trans_importance = sorted(
                [{"feature": str(n), "importance": float(imp)} for n, imp in zip(transformed_names, importances)],
                key=lambda x: x["importance"], reverse=True
            )

            orig_importance = {}
            sorted_cols = sorted(X.columns, key=lambda v: len(str(v)), reverse=True)

            for name, importance in zip(transformed_names, importances):
                cleaned = str(name).replace("numeric__", "").replace("categorical__", "")
                orig_name = cleaned
                for col in sorted_cols:
                    col_str = str(col)
                    if cleaned == col_str or any(cleaned.startswith(col_str + sep) for sep in ["_", "=", "["]):
                        orig_name = col_str
                        break
                orig_importance[orig_name] = orig_importance.get(orig_name, 0.0) + float(importance)

            orig_sorted = [
                {"feature": str(f), "importance": float(imp)} 
                for f, imp in sorted(orig_importance.items(), key=lambda x: x[1], reverse=True)
            ]

            total_imp = sum(item["importance"] for item in orig_sorted)
            for item in orig_sorted:
                item["importance_percentage"] = float(item["importance"] / total_imp * 100) if total_imp > 0 else 0.0

            top_features = orig_sorted[:top_n]

            # --- Visualization ---
            plot_path = self.plots_dir / filename
            fig, ax = plt.subplots(figsize=figsize)

            if top_features:
                plot_feats = list(reversed(top_features))
                labels = [item["feature"] for item in plot_feats]
                values = [item["importance"] for item in plot_feats]

                bars = ax.barh(labels, values)
                ax.set_title("Random Forest Feature Importance")
                ax.set_xlabel("Mean Decrease in Impurity")
                ax.set_ylabel("Feature")

                for bar, val in zip(bars, values):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{val:.4f}",
                            va="center", ha="left", fontsize=8)
                ax.grid(axis="x", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- Result Compilation ---
            result = {
                "component": comp_name,
                "status": "success",
                "target": self.target,
                "target_type": self.target_type,
                "n_observations": int(len(X)),
                "n_estimators": int(n_estimators),
                "max_depth": int(max_depth) if max_depth is not None else None,
                "min_samples_split": int(min_samples_split),
                "min_samples_leaf": int(min_samples_leaf),
                "max_features": str(max_features) if max_features is not None else None,
                "class_weight": str(class_weight) if self.target_type == "classification" else None,
                "transformed_feature_count": int(len(transformed_names)),
                "original_feature_count": int(len(orig_sorted)),
                "transformed_feature_importance": trans_importance,
                "original_feature_importance": orig_sorted,
                "top_features": top_features,
                "plot": str(plot_path),
            }

            self.results[comp_name] = result
            self.component_status[comp_name] = {"status": "success", "plot": str(plot_path)}
            return result

        except Exception as exc:
            error_result = {
                "component": comp_name,
                "status": "error",
                "target": self.target,
                "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results[comp_name] = error_result
            self.component_status[comp_name] = {"status": "error", "error": str(exc)}
            return error_result


    # ========================================================
    # COMPONENT 7: PERMUTATION IMPORTANCE
    # ========================================================

    def permutation_importance(
        self,
        test_size: float = 0.20,
        n_repeats: int = 10,
        scoring: str | None = None,
        top_n: int = 20,
        filename: str = "permutation_importance.png",
        figsize: tuple[int, int] = (12, 8),
    ) -> dict[str, Any]:
        """Calculate permutation importance for the Random Forest."""
        

        component_name = "permutation_importance"

        try:
            # --- Validation ---
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1.")
            if n_repeats <= 0:
                raise ValueError("n_repeats must be greater than 0.")
            if top_n <= 0:
                raise ValueError("top_n must be greater than 0.")

            # --- Data & Target Preparation ---
            X = self.X.copy()
            valid_target_mask = self.y.notna()
            X, y = X.loc[valid_target_mask], self.y.loc[valid_target_mask]

            if self.target_type == "classification":
                y_model = y.copy()
                stratify = y_model
                if y_model.nunique() < 2:
                    raise ValueError("Classification target must contain at least two classes.")
                default_scoring = "accuracy"
            elif self.target_type == "regression":
                y_model = pd.to_numeric(y, errors="coerce")
                numeric_mask = y_model.notna()
                X, y_model = X.loc[numeric_mask], y_model.loc[numeric_mask]
                if len(y_model) < 3:
                    raise ValueError("Not enough valid observations for regression permutation importance.")
                stratify = None
                default_scoring = "r2"
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")

            scoring = scoring or default_scoring

            # --- Train / Validation Split ---
            try:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y_model, test_size=test_size, random_state=self.random_state, stratify=stratify
                )
            except ValueError as exc:
                if self.target_type == "classification":
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        X, y_model, test_size=test_size, random_state=self.random_state, stratify=None
                    )
                else:
                    raise exc

            # --- Get or Build Model Pipeline ---
            if hasattr(self, "random_forest_model"):
                model_pipeline = clone(self.random_forest_model)
            else:
                numeric_features = [f for f in self.numeric_features if f in X.columns]
                categorical_features = [f for f in self.categorical_features if f in X.columns]
                transformers = []

                if numeric_features:
                    transformers.append(("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
                if categorical_features:
                    cat_pipe = Pipeline([
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                    ])
                    transformers.append(("categorical", cat_pipe, categorical_features))

                if not transformers:
                    raise ValueError("No numeric or categorical features are available.")

                preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
                model = (
                    RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=self.random_state, n_jobs=-1)
                    if self.target_type == "classification"
                    else RandomForestRegressor(n_estimators=300, random_state=self.random_state, n_jobs=-1)
                )
                model_pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

            # --- Fit & Score ---
            model_pipeline.fit(X_train, y_train)
            self.permutation_model = model_pipeline
            baseline_score = model_pipeline.score(X_valid, y_valid)

            perm_result = sklearn_permutation_importance(
                model_pipeline, X_valid, y_valid, scoring=scoring,
                n_repeats=n_repeats, random_state=self.random_state, n_jobs=-1
            )

            # --- Build Feature Results ---
            col_list = list(X_valid.columns)
            feature_importance = []
            
            for idx, feature in enumerate(col_list):
                importances_arr = perm_result.importances[idx]
                feature_importance.append({
                    "feature": str(feature),
                    "importance_mean": float(perm_result.importances_mean[idx]),
                    "importance_std": float(perm_result.importances_std[idx]),
                    "importance_min": float(importances_arr.min()),
                    "importance_max": float(importances_arr.max()),
                })

            feature_importance.sort(key=lambda item: item["importance_mean"], reverse=True)
            top_features = feature_importance[:top_n]

            # --- Visualization ---
            plot_path = self.plots_dir / filename
            fig, ax = plt.subplots(figsize=figsize)

            if top_features:
                plot_features = list(reversed(top_features))
                labels = [item["feature"] for item in plot_features]
                means = [item["importance_mean"] for item in plot_features]
                stds = [item["importance_std"] for item in plot_features]

                ax.barh(labels, means, xerr=stds, capsize=3)
                ax.axvline(0, color="black", linewidth=1)
                ax.set_title("Permutation Importance")
                ax.set_xlabel(f"Mean importance ({scoring})")
                ax.set_ylabel("Feature")
                ax.grid(axis="x", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "No permutation importance available", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- Compile & Store Result ---
            result = {
                "component": component_name,
                "status": "success",
                "target": self.target,
                "target_type": self.target_type,
                "scoring": scoring,
                "test_size": float(test_size),
                "n_repeats": int(n_repeats),
                "training_rows": int(len(X_train)),
                "validation_rows": int(len(X_valid)),
                "baseline_score": float(baseline_score),
                "feature_importance": feature_importance,
                "top_features": top_features,
                "plot": str(plot_path),
            }

            self.results[component_name] = result
            self.component_status[component_name] = {"status": "success", "plot": str(plot_path)}
            return result

        except Exception as exc:
            error_result = {
                "component": component_name,
                "status": "error",
                "target": getattr(self, "target", None),
                "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results[component_name] = error_result
            self.component_status[component_name] = {"status": "error", "error": str(exc)}
            return error_result


    # ========================================================
    # COMPONENT 8: CROSS VALIDATION (Optimized)
    # ========================================================

    def cross_validation(
        self,
        cv: int = 5,
        n_estimators: int = 300,
        scoring: list[str] | None = None,
        shuffle: bool = True,
        top_n: int | None = None,
        filename: str = "cross_validated_metrics.png",
        figsize: tuple[int, int] = (12, 7),
    ) -> dict[str, Any]:
        """
        Perform leakage-safe cross-validation using a
        preprocessing + Random Forest pipeline.
        """
        

        component_name = "cross_validation"

        try:
            # =================================================
            # VALIDATION
            # =================================================
            if cv < 2:
                raise ValueError("cv must be at least 2.")

            if n_estimators <= 0:
                raise ValueError("n_estimators must be greater than 0.")

            # =================================================
            # PREPARE DATA
            # =================================================
            X = self.X.copy()
            valid_target_mask = self.y.notna()

            X = X.loc[valid_target_mask]
            y = self.y.loc[valid_target_mask]

            if len(X) < cv:
                raise ValueError(
                    f"Dataset contains only {len(X)} valid observations, but cv={cv}."
                )

            # =================================================
            # TARGET PREPARATION
            # =================================================
            if self.target_type == "classification":
                y_model = y.copy()

                if y_model.nunique() < 2:
                    raise ValueError("Classification target must contain at least two classes.")

                class_counts = y_model.value_counts()
                min_class_count = int(class_counts.min())

                if min_class_count < cv:
                    raise ValueError(
                        f"The smallest target class contains only {min_class_count} observations, "
                        f"but cv={cv}. Stratified cross-validation requires at least cv observations in every class."
                    )

                if scoring is None:
                    scoring = [
                        "accuracy",
                        "precision_weighted",
                        "recall_weighted",
                        "f1_weighted",
                    ]
                    if y_model.nunique() == 2:
                        scoring.append("roc_auc")

                splitter = StratifiedKFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=self.random_state if shuffle else None,
                )

                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    class_weight="balanced",
                    random_state=self.random_state,
                    n_jobs=-1,
                )

            elif self.target_type == "regression":
                y_model = pd.to_numeric(y, errors="coerce")
                numeric_mask = y_model.notna()

                X = X.loc[numeric_mask]
                y_model = y_model.loc[numeric_mask]

                if len(X) < cv:
                    raise ValueError(
                        f"Only {len(X)} valid regression observations are available for cv={cv}."
                    )

                if scoring is None:
                    scoring = [
                        "r2",
                        "neg_mean_absolute_error",
                        "neg_mean_squared_error",
                    ]

                splitter = KFold(
                    n_splits=cv,
                    shuffle=shuffle,
                    random_state=self.random_state if shuffle else None,
                )

                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=self.random_state,
                    n_jobs=-1,
                )

            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")

            # =================================================
            # FEATURE TYPES & PREPROCESSING PIPELINE
            # =================================================
            numeric_features = [f for f in self.numeric_features if f in X.columns]
            categorical_features = [f for f in self.categorical_features if f in X.columns]

            transformers = []

            if numeric_features:
                numeric_pipeline = Pipeline(
                    steps=[("imputer", SimpleImputer(strategy="median"))]
                )
                transformers.append(("numeric", numeric_pipeline, numeric_features))

            if categorical_features:
                categorical_pipeline = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                )
                transformers.append(("categorical", categorical_pipeline, categorical_features))

            if not transformers:
                raise ValueError("No numeric or categorical features are available for cross-validation.")

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

            # =================================================
            # CROSS VALIDATION EXECUTION
            # =================================================
            cv_results = cross_validate(
                pipeline,
                X,
                y_model,
                cv=splitter,
                scoring=scoring,
                return_train_score=True,
                return_estimator=True,
                n_jobs=1,
                error_score="raise",
            )

            # =================================================
            # BUILD METRIC RESULTS & AGGREGATES
            # =================================================
            metric_results = {}
            for key, values in cv_results.items():
                if key.startswith("test_") or key.startswith("train_") or key in {"fit_time", "score_time"}:
                    metric_results[key] = [float(val) for val in values]

            aggregate_metrics = {}
            for key, values in metric_results.items():
                arr = np.asarray(values, dtype=float)
                aggregate_metrics[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }

            # User-friendly regression metrics conversion (MAE & RMSE)
            if self.target_type == "regression":
                for neg_metric, friendly_name in [
                    ("test_neg_mean_absolute_error", "test_mae"),
                    ("test_neg_mean_squared_error", "test_rmse")
                ]:
                    if neg_metric in metric_results:
                        raw_vals = np.abs(np.asarray(metric_results[neg_metric], dtype=float))
                        if "squared" in neg_metric:
                            raw_vals = np.sqrt(raw_vals)
                        aggregate_metrics[friendly_name] = {
                            "mean": float(raw_vals.mean()),
                            "std": float(raw_vals.std(ddof=1)) if len(raw_vals) > 1 else 0.0,
                            "min": float(raw_vals.min()),
                            "max": float(raw_vals.max()),
                        }

            metric_notes = {}
            if self.target_type == "classification":
                metric_notes = {
                    "precision": "Weighted precision is used.",
                    "recall": "Weighted recall is used.",
                    "f1": "Weighted F1 is used.",
                }
                if scoring and "roc_auc" in scoring:
                    metric_notes["roc_auc"] = "ROC-AUC is calculated for binary classification."

            # =================================================
            # VISUALIZATION
            # =================================================
            plot_path = self.plots_dir / filename
            test_metrics = {
                k: v for k, v in metric_results.items()
                if k.startswith("test_") and k not in {"test_neg_mean_absolute_error", "test_neg_mean_squared_error"}
            }

            if self.target_type == "regression":
                if "test_r2" in metric_results:
                    test_metrics["test_r2"] = metric_results["test_r2"]
                if "test_neg_mean_absolute_error" in metric_results:
                    test_metrics["test_mae"] = [abs(v) for v in metric_results["test_neg_mean_absolute_error"]]
                if "test_neg_mean_squared_error" in metric_results:
                    test_metrics["test_rmse"] = [np.sqrt(abs(v)) for v in metric_results["test_neg_mean_squared_error"]]

            fig, ax = plt.subplots(figsize=figsize)

            if test_metrics:
                labels, means, stds = [], [], []
                for m_name, vals in test_metrics.items():
                    arr = np.asarray(vals, dtype=float)
                    labels.append(m_name.replace("test_", ""))
                    means.append(float(arr.mean()))
                    stds.append(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0)

                x_pos = np.arange(len(labels))
                bars = ax.bar(x_pos, means, yerr=stds, capsize=4)

                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylabel("Cross-Validation Score")
                ax.set_title(f"{cv}-Fold Cross-Validation")
                ax.grid(axis="y", alpha=0.25)

                for bar, mean in zip(bars, means):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{mean:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )
            else:
                ax.text(0.5, 0.5, "No validation metrics available", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # =================================================
            # STORAGE & RETURN
            # =================================================
            self.cross_validation_estimators = cv_results["estimator"]
            self.cross_validation_results = cv_results

            result = {
                "component": component_name,
                "status": "success",
                "target": self.target,
                "target_type": self.target_type,
                "cv_folds": int(cv),
                "shuffle": bool(shuffle),
                "n_observations": int(len(X)),
                "n_estimators": int(n_estimators),
                "scoring": [str(m) for m in scoring],
                "fold_metrics": metric_results,
                "aggregate_metrics": aggregate_metrics,
                "metric_notes": metric_notes,
                "plot": str(plot_path),
            }

            self.results[component_name] = result
            self.component_status[component_name] = {
                "status": "success",
                "plot": str(plot_path),
                "cv_folds": int(cv),
            }

            return result

        except Exception as exc:
            error_result = {
                "component": component_name,
                "status": "error",
                "target": getattr(self, "target", None),
                "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results[component_name] = error_result
            self.component_status[component_name] = {
                "status": "error",
                "error": str(exc),
            }
            return error_result


    # ========================================================
    # COMPONENT 9: LOGISTIC REGRESSION
    # ========================================================

    def logistic_regression(
        self,
        test_size: float = 0.20,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        top_n: int = 20,
        filename: str = "logistic_regression.png",
        figsize: tuple[int, int] = (12, 8),
    ) -> dict[str, Any]:
        """
        Train and evaluate a Logistic Regression model.

        This component is applicable only to classification targets.
        """
    

        component_name = "logistic_regression"

        try:
            # TARGET TYPE CHECK
            if self.target_type != "classification":
                result = {
                    "component": component_name,
                    "status": "skipped",
                    "target": self.target,
                    "target_type": self.target_type,
                    "reason": "Logistic Regression requires a classification target.",
                }
                self.results[component_name] = result
                self.component_status[component_name] = {"status": "skipped", "reason": result["reason"]}
                return result

            # VALIDATION
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1.")
            if C <= 0:
                raise ValueError("C must be greater than 0.")
            if max_iter <= 0:
                raise ValueError("max_iter must be greater than 0.")
            if top_n <= 0:
                raise ValueError("top_n must be greater than 0.")

            # PREPARE DATA
            X = self.X.copy()
            valid_target_mask = self.y.notna()
            X, y = X.loc[valid_target_mask], self.y.loc[valid_target_mask]

            if len(X) < 3:
                raise ValueError("Not enough observations for Logistic Regression.")
            if y.nunique() < 2:
                raise ValueError("Classification target must contain at least two classes.")

            # TRAIN / VALIDATION SPLIT
            try:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state, stratify=y
                )
                stratified_split = True
            except ValueError:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y, test_size=test_size, random_state=self.random_state, stratify=None
                )
                stratified_split = False

            # FEATURE TYPES & PIPELINES
            numeric_features = [f for f in self.numeric_features if f in X.columns]
            categorical_features = [f for f in self.categorical_features if f in X.columns]
            transformers = []

            if numeric_features:
                numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
                transformers.append(("numeric", numeric_pipeline, numeric_features))

            if categorical_features:
                categorical_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ])
                transformers.append(("categorical", categorical_pipeline, categorical_features))

            if not transformers:
                raise ValueError("No numeric or categorical features are available.")

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=self.random_state)
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

            # FIT
            pipeline.fit(X_train, y_train)
            self.logistic_regression_model = pipeline

            # PREDICTIONS & METRICS
            y_pred = pipeline.predict(X_valid)
            y_proba = None
            try:
                y_proba = pipeline.predict_proba(X_valid)
            except Exception:
                pass

            accuracy = accuracy_score(y_valid, y_pred)
            precision = precision_score(y_valid, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_valid, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_valid, y_pred, average="weighted", zero_division=0)
            
            roc_auc = None
            if y_proba is not None and y.nunique() == 2:
                try:
                    roc_auc = roc_auc_score(y_valid, y_proba[:, 1])
                except Exception:
                    pass

            # COEFFICIENTS & FEATURE IMPORTANCE
            fitted_preprocessor = pipeline.named_steps["preprocessor"]
            fitted_model = pipeline.named_steps["model"]

            try:
                transformed_names = fitted_preprocessor.get_feature_names_out().tolist()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(len(fitted_model.coef_[0]))]

            coefficients = fitted_model.coef_
            transformed_coefficients = []
            original_importance = {}

            # Binary classification
            if coefficients.shape[0] == 1:
                for name, coef in zip(transformed_names, coefficients[0]):
                    transformed_coefficients.append({
                        "feature": str(name),
                        "coefficient": float(coef),
                        "absolute_coefficient": float(abs(coef)),
                    })
                transformed_coefficients.sort(key=lambda x: x["absolute_coefficient"], reverse=True)
            # Multiclass classification
            else:
                for f_idx, name in enumerate(transformed_names):
                    class_coefs = {}
                    for c_idx, c_label in enumerate(fitted_model.classes_):
                        coef = coefficients[c_idx, f_idx]
                        class_coefs[str(c_label)] = {
                            "coefficient": float(coef),
                            "absolute_coefficient": float(abs(coef)),
                        }
                    max_abs = max(v["absolute_coefficient"] for v in class_coefs.values())
                    transformed_coefficients.append({
                        "feature": str(name),
                        "classes": class_coefs,
                        "importance": float(max_abs),
                    })
                transformed_coefficients.sort(key=lambda x: x["importance"], reverse=True)

            # Original Feature Aggregation
            for idx, transformed_name in enumerate(transformed_names):
                cleaned_name = str(transformed_name).replace("numeric__", "").replace("categorical__", "")
                original_name = cleaned_name

                for col in sorted(X.columns, key=lambda v: len(str(v)), reverse=True):
                    col_str = str(col)
                    if cleaned_name == col_str or cleaned_name.startswith((col_str + "_", col_str + "=", col_str + "[")):
                        original_name = col_str
                        break

                if coefficients.shape[0] == 1:
                    magnitude = abs(float(coefficients[0, idx]))
                else:
                    magnitude = max(abs(float(coefficients[c_idx, idx])) for c_idx in range(coefficients.shape[0]))

                original_importance[original_name] = original_importance.get(original_name, 0.0) + magnitude

            original_importance_sorted = sorted(
                [{"feature": str(f), "importance": float(imp)} for f, imp in original_importance.items()],
                key=lambda x: x["importance"],
                reverse=True,
            )
            top_features = original_importance_sorted[:top_n]

            # VISUALIZATION
            plot_path = self.plots_dir / filename
            fig, ax = plt.subplots(figsize=figsize)

            if top_features:
                plot_features = list(reversed(top_features))
                ax.barh([i["feature"] for i in plot_features], [i["importance"] for i in plot_features])
                ax.set_title("Logistic Regression Feature Importance")
                ax.set_xlabel("Absolute Coefficient Magnitude")
                ax.set_ylabel("Feature")
                ax.grid(axis="x", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "No feature coefficients available", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # RESULT DICT
            result = {
                "component": component_name,
                "status": "success",
                "target": self.target,
                "target_type": self.target_type,
                "test_size": float(test_size),
                "stratified_split": bool(stratified_split),
                "training_rows": int(len(X_train)),
                "validation_rows": int(len(X_valid)),
                "C": float(C),
                "max_iter": int(max_iter),
                "solver": str(solver),
                "classes": [str(v) for v in fitted_model.classes_],
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision_weighted": float(precision),
                    "recall_weighted": float(recall),
                    "f1_weighted": float(f1),
                    "roc_auc": float(roc_auc) if roc_auc is not None else None,
                },
                "transformed_feature_count": int(len(transformed_names)),
                "transformed_coefficients": transformed_coefficients,
                "original_feature_importance": original_importance_sorted,
                "top_features": top_features,
                "plot": str(plot_path),
            }

            self.results[component_name] = result
            self.component_status[component_name] = {"status": "success", "plot": str(plot_path)}
            return result

        except Exception as exc:
            error_result = {
                "component": component_name,
                "status": "error",
                "target": getattr(self, "target", None),
                "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results[component_name] = error_result
            self.component_status[component_name] = {"status": "error", "error": str(exc)}
            return error_result


    def decision_tree(
        self,
        test_size: float = 0.20,
        max_depth: int | None = 5,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str | None = None,
        top_n: int = 20,
        filename: str = "decision_tree.png",
        figsize: tuple[int, int] = (18, 10),
    ) -> dict[str, Any]:
        """Train and evaluate a Decision Tree (Classification or Regression)."""
        

        component_name = "decision_tree"

        try:
            # =================================================
            # VALIDATION
            # =================================================
            if not 0 < test_size < 1:
                raise ValueError("test_size must be between 0 and 1.")
            if max_depth is not None and max_depth <= 0:
                raise ValueError("max_depth must be greater than 0 or None.")
            if min_samples_split < 2:
                raise ValueError("min_samples_split must be at least 2.")
            if min_samples_leaf < 1:
                raise ValueError("min_samples_leaf must be at least 1.")
            if top_n <= 0:
                raise ValueError("top_n must be greater than 0.")

            # =================================================
            # PREPARE DATA
            # =================================================
            X = self.X.copy()
            valid_target_mask = self.y.notna()
            X, y = X.loc[valid_target_mask], self.y.loc[valid_target_mask]

            if len(X) < 3:
                raise ValueError("Not enough observations for Decision Tree.")

            # =================================================
            # TARGET PREPARATION
            # =================================================
            if self.target_type == "classification":
                y_model = y.copy()
                if y_model.nunique() < 2:
                    raise ValueError("Classification target must contain at least two classes.")
                
                criterion_used = criterion if criterion is not None else "gini"
                valid_criteria = {"gini", "entropy", "log_loss"}
                if criterion_used not in valid_criteria:
                    raise ValueError(f"Invalid classification criterion. Choose from: {sorted(valid_criteria)}")

                model = DecisionTreeClassifier(
                    criterion=criterion_used, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state
                )

            elif self.target_type == "regression":
                y_model = pd.to_numeric(y, errors="coerce")
                numeric_target_mask = y_model.notna()
                X, y_model = X.loc[numeric_target_mask], y_model.loc[numeric_target_mask]

                if len(X) < 3:
                    raise ValueError("Not enough valid observations for Decision Tree regression.")

                criterion_used = criterion if criterion is not None else "squared_error"
                valid_criteria = {"squared_error", "absolute_error", "poisson"}
                if criterion_used not in valid_criteria:
                    raise ValueError(f"Invalid regression criterion. Choose from: {sorted(valid_criteria)}")

                if criterion_used == "poisson" and (y_model < 0).any():
                    raise ValueError("The Poisson criterion requires non-negative target values.")

                model = DecisionTreeRegressor(
                    criterion=criterion_used, max_depth=max_depth,
                    min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                    random_state=self.random_state
                )
            else:
                raise ValueError(f"Unsupported target type: {self.target_type}")

            # =================================================
            # TRAIN / VALIDATION SPLIT
            # =================================================
            if self.target_type == "classification":
                try:
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        X, y_model, test_size=test_size, random_state=self.random_state, stratify=y_model
                    )
                    stratified_split = True
                except ValueError:
                    X_train, X_valid, y_train, y_valid = train_test_split(
                        X, y_model, test_size=test_size, random_state=self.random_state, stratify=None
                    )
                    stratified_split = False
            else:
                X_train, X_valid, y_train, y_valid = train_test_split(
                    X, y_model, test_size=test_size, random_state=self.random_state
                )
                stratified_split = False

            # =================================================
            # FEATURE TYPES & PIPELINES
            # =================================================
            numeric_features = [f for f in self.numeric_features if f in X.columns]
            categorical_features = [f for f in self.categorical_features if f in X.columns]
            transformers = []

            if numeric_features:
                numeric_pipeline = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])
                transformers.append(("numeric", numeric_pipeline, numeric_features))

            if categorical_features:
                categorical_pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ])
                transformers.append(("categorical", categorical_pipeline, categorical_features))

            if not transformers:
                raise ValueError("No numeric or categorical features are available.")

            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

            # =================================================
            # FIT & PREDICT
            # =================================================
            pipeline.fit(X_train, y_train)
            self.decision_tree_model = pipeline
            y_pred = pipeline.predict(X_valid)

            # =================================================
            # METRICS
            # =================================================
            if self.target_type == "classification":
                accuracy = accuracy_score(y_valid, y_pred)
                precision = precision_score(y_valid, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_valid, y_pred, average="weighted", zero_division=0)
                f1 = f1_score(y_valid, y_pred, average="weighted", zero_division=0)
                roc_auc = None

                try:
                    y_proba = pipeline.predict_proba(X_valid)
                    if y_model.nunique() == 2:
                        roc_auc = roc_auc_score(y_valid, y_proba[:, 1])
                except Exception:
                    roc_auc = None

                metrics = {
                    "accuracy": float(accuracy),
                    "precision_weighted": float(precision),
                    "recall_weighted": float(recall),
                    "f1_weighted": float(f1),
                    "roc_auc": float(roc_auc) if roc_auc is not None else None,
                }
            else:
                r2 = r2_score(y_valid, y_pred)
                mae = mean_absolute_error(y_valid, y_pred)
                mse = mean_squared_error(y_valid, y_pred)
                rmse = np.sqrt(mse)
                metrics = {"r2": float(r2), "mae": float(mae), "mse": float(mse), "rmse": float(rmse)}

            # =================================================
            # FITTED COMPONENTS & FEATURE IMPORTANCE
            # =================================================
            fitted_preprocessor = pipeline.named_steps["preprocessor"]
            fitted_model = pipeline.named_steps["model"]

            try:
                transformed_names = fitted_preprocessor.get_feature_names_out().tolist()
            except Exception:
                transformed_names = [f"feature_{i}" for i in range(len(fitted_model.feature_importances_))]

            transformed_importance = sorted(
                [{"feature": str(name), "importance": float(imp)} 
                 for name, imp in zip(transformed_names, fitted_model.feature_importances_)],
                key=lambda x: x["importance"], reverse=True
            )

            original_importance = {}
            for index, transformed_name in enumerate(transformed_names):
                cleaned_name = str(transformed_name).replace("numeric__", "").replace("categorical__", "")
                original_name = cleaned_name

                for column in sorted(X.columns, key=lambda v: len(str(v)), reverse=True):
                    column = str(column)
                    if cleaned_name == column or cleaned_name.startswith(f"{column}_") or \
                       cleaned_name.startswith(f"{column}=") or cleaned_name.startswith(f"{column}["):
                        original_name = column
                        break

                original_importance[original_name] = original_importance.get(original_name, 0.0) + float(fitted_model.feature_importances_[index])

            original_importance_sorted = sorted(
                [{"feature": str(f), "importance": float(imp)} for f, imp in original_importance.items()],
                key=lambda x: x["importance"], reverse=True
            )
            top_features = original_importance_sorted[:top_n]
            tree_feature_names = [str(name) for name in transformed_names]
            class_names = [str(v) for v in fitted_model.classes_] if self.target_type == "classification" else None

            tree_depth = int(fitted_model.get_depth())
            tree_leaves = int(fitted_model.get_n_leaves())
            tree_nodes = int(fitted_model.tree_.node_count)

            # =================================================
            # VISUALIZATIONS
            # =================================================
            tree_plot_path = self.plots_dir / filename
            importance_filename = filename.replace(".png", "_importance.png")
            importance_plot_path = self.plots_dir / importance_filename

            display_depth = max_depth if max_depth is not None else min(tree_depth, 6)

            # Tree Plot
            fig, ax = plt.subplots(figsize=figsize)
            plot_tree(
                fitted_model, feature_names=tree_feature_names, class_names=class_names,
                filled=True, rounded=True, proportion=False, precision=2,
                max_depth=display_depth, fontsize=8, ax=ax
            )
            ax.set_title("Decision Tree")
            plt.tight_layout()
            fig.savefig(tree_plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Feature Importance Plot
            fig, ax = plt.subplots(figsize=(12, 8))
            if top_features:
                plot_features = list(reversed(top_features))
                labels = [item["feature"] for item in plot_features]
                values = [item["importance"] for item in plot_features]
                ax.barh(labels, values)
                ax.set_title("Decision Tree Feature Importance")
                ax.set_xlabel("Impurity-Based Importance")
                ax.set_ylabel("Feature")
                ax.grid(axis="x", alpha=0.25)
            else:
                ax.text(0.5, 0.5, "No feature importance available", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            fig.savefig(importance_plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # =================================================
            # RESULT
            # =================================================
            result = {
                "component": component_name, "status": "success", "target": self.target,
                "target_type": self.target_type, "test_size": float(test_size),
                "stratified_split": bool(stratified_split), "training_rows": int(len(X_train)),
                "validation_rows": int(len(X_valid)), "criterion": str(criterion_used),
                "max_depth": int(max_depth) if max_depth is not None else None,
                "min_samples_split": int(min_samples_split), "min_samples_leaf": int(min_samples_leaf),
                "metrics": metrics, "tree_depth": tree_depth, "tree_leaves": tree_leaves,
                "tree_nodes": tree_nodes, "transformed_feature_count": int(len(transformed_names)),
                "transformed_feature_importance": transformed_importance,
                "original_feature_importance": original_importance_sorted, "top_features": top_features,
                "tree_plot": str(tree_plot_path), "importance_plot": str(importance_plot_path),
            }

            self.results[component_name] = result
            self.component_status[component_name] = {
                "status": "success", "tree_plot": str(tree_plot_path), "importance_plot": str(importance_plot_path)
            }
            return result

        except Exception as exc:
            error_result = {
                "component": component_name, "status": "error",
                "target": getattr(self, "target", None), "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results[component_name] = error_result
            self.component_status[component_name] = {"status": "error", "error": str(exc)}
            return error_result


    # ========================================================
    # COMPONENT 11: IV / WOE
    # ========================================================

    def iv_woe(
        self,
        bins: int = 10,
        min_bin_size: float = 0.05,
        smoothing: float = 0.5,
        top_n: int = 20,
        filename: str = "iv_woe.png",
        figsize: tuple[int, int] = (14, 9),
    ) -> dict[str, Any]:
        """
        Calculate Information Value (IV) and Weight of
        Evidence (WOE) for a binary classification target.

        Numeric features
        ----------------
        Numeric variables are binned using quantile-based
        binning.

        Categorical features
        --------------------
        Categorical variables are grouped by category.

        Missing values
        --------------
        Missing values are retained as a separate bin.

        WOE convention
        --------------
        WOE = ln(distribution_non_event /
                 distribution_event)

        IV convention
        --------------
        IV = sum(
            (distribution_non_event -
             distribution_event) * WOE
        )

        Target convention
        -----------------
        The first sorted target class is treated as:

            non-event

        The second sorted target class is treated as:

            event

        This convention is explicitly returned in the result.

        Parameters
        ----------
        bins:
            Number of quantile bins for numeric variables.

        min_bin_size:
            Minimum proportion of observations required for
            a numeric bin.

        smoothing:
            Additive smoothing applied to event/non-event
            counts to prevent log(0).

        top_n:
            Number of features displayed in the IV plot.

        filename:
            Name of the saved visualization.

        figsize:
            Matplotlib figure size.

        Returns
        -------
        dict
            JSON-compatible IV/WOE results.
        """


        c_name = "iv_woe"

        try:
            # Target Type Check
            if self.target_type != "classification":
                res = {
                    "component": c_name,
                    "status": "skipped",
                    "target": self.target,
                    "target_type": self.target_type,
                    "reason": "IV/WOE is implemented for binary classification targets.",
                }
                self.results[c_name] = res
                self.component_status[c_name] = {"status": "skipped", "reason": res["reason"]}
                return res

            # Prepare Target
            y = self.y.copy()
            valid_mask = y.notna()
            X = self.X.loc[valid_mask].copy()
            y = y.loc[valid_mask].copy()
            unique_classes = list(y.dropna().unique())

            if len(unique_classes) != 2:
                res = {
                    "component": c_name,
                    "status": "skipped",
                    "target": self.target,
                    "target_type": self.target_type,
                    "reason": f"IV/WOE requires exactly two target classes. Found {len(unique_classes)}.",
                    "target_classes": [str(v) for v in unique_classes],
                }
                self.results[c_name] = res
                self.component_status[c_name] = {"status": "skipped", "reason": res["reason"]}
                return res

            # Validation
            if bins < 2:
                raise ValueError("bins must be at least 2.")
            if not 0 < min_bin_size < 1:
                raise ValueError("min_bin_size must be between 0 and 1.")
            if smoothing <= 0:
                raise ValueError("smoothing must be greater than 0.")
            if top_n <= 0:
                raise ValueError("top_n must be greater than 0.")

            # Target Class Convention
            sorted_classes = sorted(unique_classes, key=str)
            non_event_class, event_class = sorted_classes[0], sorted_classes[1]

            total_events = int((y == event_class).sum())
            total_non_events = int((y == non_event_class).sum())

            if total_events == 0:
                raise ValueError("No event observations found.")
            if total_non_events == 0:
                raise ValueError("No non-event observations found.")

            # Helper: Numeric Binning
            def create_numeric_bins(series: pd.Series) -> pd.Series:
                num_s = pd.to_numeric(series, errors="coerce")
                res_s = pd.Series(index=series.index, dtype="object")
                missing = num_s.isna()
                res_s.loc[missing] = "__MISSING__"
                valid = num_s.loc[~missing]

                if valid.empty:
                    return res_s
                if valid.nunique() <= 1:
                    res_s.loc[~missing] = "ALL_VALUES"
                    return res_s

                try:
                    res_s.loc[~missing] = pd.qcut(valid, q=bins, duplicates="drop").astype(str)
                except (ValueError, TypeError):
                    try:
                        res_s.loc[~missing] = pd.cut(valid, bins=bins, duplicates="drop").astype(str)
                    except (ValueError, TypeError):
                        res_s.loc[~missing] = "ALL_VALUES"
                return res_s

            # Helper: Categorical Grouping
            def create_categorical_bins(series: pd.Series) -> pd.Series:
                res_s = series.astype("object").copy()
                missing = res_s.isna()
                res_s.loc[missing] = "__MISSING__"
                res_s.loc[~missing] = res_s.loc[~missing].astype(str)
                return res_s

            # Helper: Calculate WOE / IV
            def calculate_woe_iv(binned_feature: pd.Series) -> tuple[float, list[dict[str, Any]]]:
                working = pd.DataFrame({"bin": binned_feature, "target": y})
                grouped = working.groupby("bin", dropna=False).agg(
                    total=("target", "size"),
                    events=("target", lambda v: int((v == event_class).sum())),
                    non_events=("target", lambda v: int((v == non_event_class).sum())),
                )

                grouped["events_smoothed"] = grouped["events"] + smoothing
                grouped["non_events_smoothed"] = grouped["non_events"] + smoothing

                s_ev_total = total_events + smoothing * len(grouped)
                s_nev_total = total_non_events + smoothing * len(grouped)

                grouped["event_distribution"] = grouped["events_smoothed"] / s_ev_total
                grouped["non_event_distribution"] = grouped["non_events_smoothed"] / s_nev_total

                grouped["woe"] = np.log(grouped["non_event_distribution"] / grouped["event_distribution"])
                grouped["iv_contribution"] = (grouped["non_event_distribution"] - grouped["event_distribution"]) * grouped["woe"]

                iv_val = float(grouped["iv_contribution"].sum())
                bin_results = [
                    {
                        "bin": str(b),
                        "total": int(r["total"]),
                        "events": int(r["events"]),
                        "non_events": int(r["non_events"]),
                        "event_distribution": float(r["event_distribution"]),
                        "non_event_distribution": float(r["non_event_distribution"]),
                        "woe": float(r["woe"]),
                        "iv_contribution": float(r["iv_contribution"]),
                    }
                    for b, r in grouped.iterrows()
                ]
                return iv_val, bin_results

            # Feature List
            candidate_features = [f for f in X.columns if f in self.numeric_features or f in self.categorical_features]
            if not candidate_features:
                raise ValueError("No numeric or categorical features are available for IV/WOE analysis.")

            # Calculate IV / WOE
            feature_results = []
            for feat in candidate_features:
                series = X[feat]
                if feat in self.numeric_features:
                    binned, f_type = create_numeric_bins(series), "numeric"
                else:
                    binned, f_type = create_categorical_bins(series), "categorical"

                proportions = binned.value_counts(normalize=True, dropna=False)
                small_bins = [[str(b)] for b, prop in proportions.items() if prop < min_bin_size]
                
                iv_val, bin_res = calculate_woe_iv(binned)
                feature_results.append({
                    "feature": str(feat),
                    "feature_type": f_type,
                    "information_value": float(iv_val),
                    "bin_count": len(bin_res),
                    "small_bins": small_bins,
                    "bins": bin_res,
                })

            # Sort by IV & Interpret
            feature_results.sort(key=lambda item: item["information_value"], reverse=True)
            top_features = feature_results[:top_n]

            def interpret_iv(val: float) -> str:
                if val < 0.02: return "very_weak"
                if val < 0.10: return "weak"
                if val < 0.30: return "medium"
                if val < 0.50: return "strong"
                return "very_strong"

            for item in feature_results:
                item["interpretation"] = interpret_iv(item["information_value"])

            # Visualization
            plot_path = self.plots_dir / filename
            fig, ax = plt.subplots(figsize=figsize)

            if top_features:
                plot_feats = list(reversed(top_features))
                labels = [item["feature"] for item in plot_feats]
                values = [item["information_value"] for item in plot_feats]

                bars = ax.barh(labels, values)
                ax.set_title("Information Value (IV)")
                ax.set_xlabel("Information Value")
                ax.set_ylabel("Feature")
                ax.grid(axis="x", alpha=0.25)

                for bar, val in zip(bars, values):
                    ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", ha="left", fontsize=9)
            else:
                ax.text(0.5, 0.5, "No IV results available", ha="center", va="center", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Result Dict
            res = {
                "component": c_name,
                "status": "success",
                "target": self.target,
                "target_type": self.target_type,
                "target_classes": [str(v) for v in sorted_classes],
                "non_event_class": str(non_event_class),
                "event_class": str(event_class),
                "total_events": total_events,
                "total_non_events": total_non_events,
                "bins": bins,
                "min_bin_size": float(min_bin_size),
                "smoothing": float(smoothing),
                "woe_formula": "ln(non_event_distribution / event_distribution)",
                "iv_formula": "sum((non_event_distribution - event_distribution) * WOE)",
                "feature_count": len(feature_results),
                "features": feature_results,
                "top_features": top_features,
                "plot": str(plot_path),
            }

            self.iv_woe_results = res
            self.results[c_name] = res
            self.component_status[c_name] = {"status": "success", "plot": str(plot_path)}
            return res

        except Exception as exc:
            err_res = {
                "component": c_name,
                "status": "error",
                "target": getattr(self, "target", None),
                "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results[c_name] = err_res
            self.component_status[c_name] = {"status": "error", "error": str(exc)}
            return err_res


    # ========================================================
    # COMPONENT 12: CLUSTERING/ K MEANS
    # ========================================================

    def kmeans(
        self,
        k: int | None = None,
        k_min: int = 2,
        k_max: int = 8,
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int | None = None,
        top_n_features: int = 10,
        filename: str = "kmeans.png",
        figsize: tuple[int, int] = (14, 10),
    ) -> dict[str, Any]:
        """Perform K-Means clustering on numeric features."""
        

        name = "kmeans"
        try:
            rs = random_state if random_state is not None else getattr(self, "random_state", 42)
            if (k is not None and k < 2) or k_min < 2:
                raise ValueError("k and k_min must be at least 2.")
            if k_max < k_min or n_init <= 0 or max_iter <= 0 or top_n_features <= 0:
                raise ValueError("Invalid parameters provided.")

            num_feats = [f for f in getattr(self, "numeric_features", []) if f in self.X.columns]
            if not num_feats:
                raise ValueError("K-Means requires at least one numeric feature.")

            X_num = self.X[num_feats].apply(pd.to_numeric, errors="coerce")
            X_num = X_num.loc[:, X_num.notna().any()]
            X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]
            X_num = X_num.loc[X_num.notna().any(axis=1)].copy()

            if X_num.shape[1] == 0 or len(X_num) < 3:
                raise ValueError("Not enough usable observations or numeric features.")

            imputer = SimpleImputer(strategy="median")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(imputer.fit_transform(X_num))

            n_samples = X_scaled.shape[0]
            max_k = min(k_max, n_samples - 1)
            if max_k < 2:
                raise ValueError("Not enough observations to evaluate multiple K-Means clusters.")

            candidate_ks = [k] if k is not None else list(range(k_min, max_k + 1))
            if not candidate_ks or (k is not None and k >= n_samples):
                raise ValueError("Invalid candidate k values.")

            eval_results = []
            for ck in candidate_ks:
                model = KMeans(n_clusters=ck, n_init=n_init, max_iter=max_iter, random_state=rs)
                labels = model.fit_predict(X_scaled)
                try:
                    sil = silhouette_score(X_scaled, labels)
                except Exception:
                    sil = None
                eval_results.append({"k": int(ck), "inertia": float(model.inertia_), "silhouette_score": float(sil) if sil is not None else None})

            selected_k = int(k) if k is not None else int(max([r for r in eval_results if r["silhouette_score"] is not None], key=lambda x: x["silhouette_score"], default=eval_results[0])["k"])

            final_model = KMeans(n_clusters=selected_k, n_init=n_init, max_iter=max_iter, random_state=rs)
            cluster_labels = final_model.fit_predict(X_scaled)
            self.kmeans_model = final_model
            self.kmeans_labels = pd.Series(cluster_labels, index=X_num.index, name="cluster")

            try:
                final_sil = silhouette_score(X_scaled, cluster_labels)
            except Exception:
                final_sil = None

            cluster_sizes = {str(int(cid)): int(cnt) for cid, cnt in pd.Series(cluster_labels).value_counts().sort_index().items()}
            scaled_centroids = final_model.cluster_centers_
            orig_centroids_arr = scaler.inverse_transform(scaled_centroids)

            original_centroids = [{"cluster": int(i), **{str(f): float(v) for f, v in zip(X_num.columns, centroid)}} for i, centroid in enumerate(orig_centroids_arr)]
            
            feature_cluster_strength = np.mean(np.abs(scaled_centroids), axis=0)
            feature_strength = sorted([{"feature": str(f), "mean_absolute_scaled_centroid": float(s)} for f, s in zip(X_num.columns, feature_cluster_strength)], key=lambda x: x["mean_absolute_scaled_centroid"], reverse=True)
            top_features = feature_strength[:top_n_features]

            if X_scaled.shape[1] >= 2:
                pca = PCA(n_components=2, random_state=rs)
                X_pca = pca.fit_transform(X_scaled)
                explained_variance = pca.explained_variance_ratio_.tolist()
            else:
                X_pca = np.column_stack([X_scaled[:, 0], np.zeros(n_samples)])
                explained_variance = [1.0, 0.0]

            plot_path = self.plots_dir / filename
            diag_path = self.plots_dir / filename.replace(".png", "_diagnostics.png")

            # Cluster Visualization
            fig, ax = plt.subplots(figsize=figsize)
            cmap = plt.get_cmap("tab10")
            for cid in sorted(set(cluster_labels)):
                mask = cluster_labels == cid
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=40, alpha=0.75, color=cmap(int(cid) % 10), label=f"Cluster {cid}")
            
            centroid_pca = pca.transform(scaled_centroids) if X_scaled.shape[1] >= 2 else np.column_stack([scaled_centroids[:, 0], np.zeros(selected_k)])
            ax.scatter(centroid_pca[:, 0], centroid_pca[:, 1], marker="X", s=180, color="black", edgecolor="white", linewidth=1.2, label="Cluster centers")
            
            if X_scaled.shape[1] >= 2:
                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.set_title("K-Means Clusters (PCA Projection)")
            else:
                ax.set_xlabel(str(X_num.columns[0]))
                ax.set_ylabel("Cluster separation")
                ax.set_title("K-Means Clusters")
            
            ax.legend(loc="best")
            ax.grid(alpha=0.20)
            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # Diagnostic Visualization
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            axes[0].plot([r["k"] for r in eval_results], [r["inertia"] for r in eval_results], marker="o")
            axes[0].axvline(selected_k, color="red", linestyle="--", alpha=0.7, label=f"Selected k={selected_k}")
            axes[0].set_title("K-Means Elbow / Inertia")
            axes[0].set_xlabel("Number of Clusters (k)")
            axes[0].set_ylabel("Inertia")
            axes[0].grid(alpha=0.25)
            axes[0].legend()

            sil_res = [r for r in eval_results if r["silhouette_score"] is not None]
            if sil_res:
                axes[1].plot([r["k"] for r in sil_res], [r["silhouette_score"] for r in sil_res], marker="o")
                axes[1].axvline(selected_k, color="red", linestyle="--", alpha=0.7, label=f"Selected k={selected_k}")
                axes[1].set_title("Silhouette Score")
                axes[1].set_xlabel("Number of Clusters (k)")
                axes[1].set_ylabel("Silhouette Score")
                axes[1].grid(alpha=0.25)
                axes[1].legend()
            else:
                axes[1].text(0.5, 0.5, "Silhouette score unavailable", ha="center", va="center", transform=axes[1].transAxes)
                axes[1].set_xticks([])
                axes[1].set_yticks([])

            plt.tight_layout()
            fig.savefig(diag_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            result = {
                "component": name,
                "status": "success",
                "features_used": list(X_num.columns),
                "feature_count": int(X_num.shape[1]),
                "observations_used": int(n_samples),
                "k": int(selected_k),
                "k_auto_selected": bool(k is None),
                "k_search_range": {"min": int(min(candidate_ks)), "max": int(max(candidate_ks))},
                "n_init": int(n_init),
                "max_iter": int(max_iter),
                "random_state": int(rs) if rs is not None else None,
                "evaluation": eval_results,
                "final_inertia": float(final_model.inertia_),
                "final_silhouette_score": float(final_sil) if final_sil is not None else None,
                "cluster_sizes": cluster_sizes,
                "centroids": original_centroids,
                "feature_cluster_strength": feature_strength,
                "top_features": top_features,
                "pca_explained_variance_ratio": [float(v) for v in explained_variance],
                "cluster_plot": str(plot_path),
                "diagnostic_plot": str(diag_path),
            }

            self.kmeans_results = result
            self.results[name] = result
            self.component_status[name] = {"status": "success", "cluster_plot": str(plot_path), "diagnostic_plot": str(diag_path)}
            return result

        except Exception as exc:
            error_result = {
                "component": name,
                "status": "error",
                "target": getattr(self, "target", None),
                "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results[name] = error_result
            self.component_status[name] = {"status": "error", "error": str(exc)}
            return error_result


    # ========================================================
    # COMPONENT 13: PCA
    # ========================================================

    def pca(
        self,
        n_components: int | float | None = None,
        max_components: int | None = None,
        top_n_features: int = 10,
        filename: str = "pca.png",
        figsize: tuple[int, int] = (14, 10),
    ) -> dict[str, Any]:
        """Perform Principal Component Analysis (PCA)."""
       

        try:
            # --- VALIDATION ---
            if max_components is not None and max_components <= 0:
                raise ValueError("max_components must be greater than 0.")
            if top_n_features <= 0:
                raise ValueError("top_n_features must be greater than 0.")
            if n_components is not None:
                if isinstance(n_components, bool):
                    raise ValueError("n_components must be an integer, float, or None.")
                if isinstance(n_components, float) and not (0 < n_components <= 1):
                    raise ValueError("A float n_components must be between 0 and 1.")
                elif isinstance(n_components, int) and n_components <= 0:
                    raise ValueError("An integer n_components must be greater than 0.")
                elif not isinstance(n_components, (int, float)):
                    raise ValueError("n_components must be an integer, float, or None.")

            # --- PREPROCESSING ---
            numeric_features = [f for f in getattr(self, "numeric_features", []) if f in self.X.columns]
            if not numeric_features:
                raise ValueError("PCA requires at least one numeric feature.")

            X_numeric = self.X[numeric_features].copy()
            for f in numeric_features:
                X_numeric[f] = pd.to_numeric(X_numeric[f], errors="coerce")

            usable_features = [f for f in X_numeric.columns if X_numeric[f].notna().any()]
            X_numeric = X_numeric[usable_features]
            if X_numeric.shape[1] == 0:
                raise ValueError("No numeric features contain usable observations.")

            variable_features = [f for f in X_numeric.columns if X_numeric[f].nunique(dropna=True) > 1]
            removed_constant_features = [f for f in X_numeric.columns if f not in variable_features]
            X_numeric = X_numeric[variable_features]
            if X_numeric.shape[1] == 0:
                raise ValueError("PCA requires at least one numeric feature with variance.")

            X_numeric = X_numeric.loc[X_numeric.notna().any(axis=1)].copy()
            if len(X_numeric) < 2:
                raise ValueError("At least two observations are required for PCA.")

            X_scaled = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(X_numeric))

            # --- FITTING & CONFIGURATION ---
            max_possible = min(X_scaled.shape[0], X_scaled.shape[1])
            if max_possible < 1:
                raise ValueError("Unable to determine a valid number of PCA components.")

            req_comp = max_possible if n_components is None else (n_components if isinstance(n_components, float) else min(n_components, max_possible))
            pca_model = PCA(n_components=req_comp)
            X_pca = pca_model.fit_transform(X_scaled)

            if max_components is not None and X_pca.shape[1] > max_components:
                actual_count = min(max_components, max_possible)
                pca_model = PCA(n_components=actual_count)
                X_pca = pca_model.fit_transform(X_scaled)

            self.pca_model = pca_model
            self.pca_scores = pd.DataFrame(X_pca, index=X_numeric.index, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])

            # --- METRICS & LOADINGS ---
            ev_ratio = pca_model.explained_variance_ratio_
            ev = pca_model.explained_variance_
            cum_var = np.cumsum(ev_ratio)

            comp_results = [{"component": f"PC{i+1}", "explained_variance": float(ev[i]), "explained_variance_ratio": float(ev_ratio[i]), "cumulative_explained_variance": float(cum_var[i])} for i in range(X_pca.shape[1])]

            loadings = pca_model.components_.T * np.sqrt(ev)
            loading_results = []
            for c_idx in range(loadings.shape[1]):
                f_rows = [{"feature": str(col), "loading": float(loadings[f_idx, c_idx]), "absolute_loading": float(abs(loadings[f_idx, c_idx]))} for f_idx, col in enumerate(X_numeric.columns)]
                f_rows.sort(key=lambda x: x["absolute_loading"], reverse=True)
                loading_results.append({"component": f"PC{c_idx+1}", "features": f_rows, "top_features": f_rows[:top_n_features]})

            var_targets = {
                f"{pct}_percent": next((i + 1 for i, c in enumerate(cum_var) if c >= pct / 100), None)
                for pct in [50, 80, 90, 95]
            }

            # --- PLOTS ---
            plot_path = self.plots_dir / filename
            var_path = self.plots_dir / filename.replace(".png", "_variance.png")
            scores_path = self.plots_dir / filename.replace(".png", "_scores.png")
            comp_nums = list(range(1, len(ev_ratio) + 1))

            # 1. Main Plot
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            axes[0].plot(comp_nums, ev_ratio, marker="o", label="Individual")
            axes[0].plot(comp_nums, cum_var, marker="s", label="Cumulative")
            axes[0].axhline(0.80, color="orange", linestyle="--", alpha=0.7, label="80%")
            axes[0].axhline(0.90, color="red", linestyle="--", alpha=0.7, label="90%")
            axes[0].set(title="PCA Explained Variance", xlabel="Principal Component", ylabel="Explained Variance Ratio", xticks=comp_nums)
            axes[0].grid(alpha=0.25)
            axes[0].legend()

            if X_pca.shape[1] >= 2:
                axes[1].scatter(X_pca[:, 0], X_pca[:, 1], s=35, alpha=0.7)
                axes[1].set(xlabel="PC1", ylabel="PC2", title="PCA 2D Projection")
                axes[1].grid(alpha=0.2)
            else:
                axes[1].text(0.5, 0.5, "Only one PCA component available", ha="center", va="center", transform=axes[1].transAxes)
                axes[1].set_xticks([])
                axes[1].set_yticks([])
            plt.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 2. Variance Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(comp_nums, ev_ratio, alpha=0.65, label="Individual variance")
            ax.plot(comp_nums, cum_var, marker="o", color="red", label="Cumulative variance")
            for y, col in zip([0.80, 0.90, 0.95], ["orange", "green", "purple"]):
                ax.axhline(y, color=col, linestyle="--", alpha=0.7)
            ax.set(title="PCA Variance Explained", xlabel="Principal Component", ylabel="Variance Ratio", xticks=comp_nums)
            ax.grid(axis="y", alpha=0.25)
            ax.legend()
            plt.tight_layout()
            fig.savefig(var_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # 3. Score Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            if X_pca.shape[1] >= 2:
                ax.scatter(X_pca[:, 0], X_pca[:, 1], s=40, alpha=0.7)
                ax.set(xlabel="PC1", ylabel="PC2", title="PCA Score Plot")
            else:
                ax.scatter(X_pca[:, 0], np.zeros(X_pca.shape[0]), s=40, alpha=0.7)
                ax.set(xlabel="PC1", ylabel="", title="PCA Score Plot")
                ax.set_yticks([])
            ax.grid(alpha=0.2)
            plt.tight_layout()
            fig.savefig(scores_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

            # --- RESULTS STRUCT ---
            result = {
                "component": "pca",
                "status": "success",
                "features_used": [str(f) for f in X_numeric.columns],
                "feature_count": int(X_numeric.shape[1]),
                "observations_used": int(X_numeric.shape[0]),
                "removed_constant_features": [str(f) for f in removed_constant_features],
                "requested_n_components": n_components,
                "max_components": int(max_components) if max_components is not None else None,
                "actual_component_count": int(X_pca.shape[1]),
                "total_explained_variance": float(ev_ratio.sum()),
                "variance_threshold_components": {
                    "50_percent": var_targets["50_percent"],
                    "80_percent": var_targets["80_percent"],
                    "90_percent": var_targets["90_percent"],
                    "95_percent": var_targets["95_percent"],
                },
                "components": comp_results,
                "loadings": loading_results,
                "plot": str(plot_path),
                "variance_plot": str(var_path),
                "scores_plot": str(scores_path),
            }

            self.pca_results = result
            self.results["pca"] = result
            self.component_status["pca"] = {"status": "success", "plot": str(plot_path), "variance_plot": str(var_path), "scores_plot": str(scores_path)}
            return result

        except Exception as exc:
            error_result = {
                "component": "pca",
                "status": "error",
                "target": getattr(self, "target", None),
                "target_type": getattr(self, "target_type", None),
                "error": str(exc),
            }
            self.results["pca"] = error_result
            self.component_status["pca"] = {"status": "error", "error": str(exc)}
            return error_result


# ========================================================
# COMPONENT 14: OUTLIER ANALYSIS
# ========================================================

def outliers(
    self,
    contamination: float = 0.05,
    z_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
    use_isolation_forest: bool = True,
    use_lof: bool = True,
    lof_neighbors: int = 20,
    n_estimators: int = 100,
    random_state: int | None = None,
    top_n_features: int = 15,
    filename: str = "outliers.png",
    figsize: tuple[int, int] = (16, 10),
) -> dict[str, Any]:
    """Perform univariate and multivariate outlier analysis.

    Methods
    -------
    1. IQR rule
    2. Z-score rule
    3. Isolation Forest
    4. Local Outlier Factor

    The target variable is not used for detecting outliers.
    """
    

    component_name = "outliers"

    try:
        if random_state is None:
            random_state = getattr(self, "random_state", 42)

        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be greater than 0 and less than or equal to 0.5.")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be greater than 0.")
        if iqr_multiplier < 0:
            raise ValueError("iqr_multiplier must be greater than or equal to 0.")
        if lof_neighbors < 1:
            raise ValueError("lof_neighbors must be at least 1.")
        if n_estimators < 1:
            raise ValueError("n_estimators must be at least 1.")
        if top_n_features < 1:
            raise ValueError("top_n_features must be at least 1.")

        numeric_features = [
            feature for feature in getattr(self, "numeric_features", [])
            if feature in self.X.columns
        ]

        if not numeric_features:
            raise ValueError("Outlier analysis requires at least one numeric feature.")

        X_numeric = self.X[numeric_features].copy()

        for feature in numeric_features:
            X_numeric[feature] = pd.to_numeric(X_numeric[feature], errors="coerce")

        usable_features = [
            feature for feature in X_numeric.columns
            if X_numeric[feature].notna().any()
        ]
        X_numeric = X_numeric[usable_features]

        if X_numeric.shape[1] == 0:
            raise ValueError("No numeric features contain usable observations.")

        usable_rows = X_numeric.notna().any(axis=1)
        X_numeric = X_numeric.loc[usable_rows].copy()

        if len(X_numeric) < 3:
            raise ValueError("At least three observations are required for outlier analysis.")

        feature_results = []
        iqr_masks = {}
        zscore_masks = {}

        for feature in X_numeric.columns:
            series = X_numeric[feature]
            non_missing = series.dropna()

            if non_missing.empty:
                continue

            q1 = float(non_missing.quantile(0.25))
            q3 = float(non_missing.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr

            iqr_mask = ((series < lower_bound) | (series > upper_bound)).fillna(False)
            iqr_masks[feature] = iqr_mask
            iqr_count = int(iqr_mask.sum())

            mean = float(non_missing.mean())
            std = float(non_missing.std(ddof=0))

            if std > 0:
                zscore = (series - mean) / std
                zscore_mask = (zscore.abs() > z_threshold).fillna(False)
                maximum_abs_zscore = float(zscore.abs().max())
            else:
                zscore = pd.Series(0.0, index=series.index)
                zscore_mask = pd.Series(False, index=series.index)
                maximum_abs_zscore = 0.0

            zscore_masks[feature] = zscore_mask
            zscore_count = int(zscore_mask.sum())
            non_missing_count = int(non_missing.shape[0])

            iqr_percentage = (
                (iqr_count / non_missing_count) * 100
                if non_missing_count else 0.0
            )
            zscore_percentage = (
                (zscore_count / non_missing_count) * 100
                if non_missing_count else 0.0
            )

            feature_results.append({
                "feature": str(feature),
                "observations": int(non_missing_count),
                "missing": int(series.isna().sum()),
                "q1": q1,
                "q3": q3,
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "mean": mean,
                "std": std,
                "iqr_outlier_count": iqr_count,
                "iqr_outlier_pct": float(iqr_percentage),
                "zscore_outlier_count": zscore_count,
                "zscore_outlier_pct": float(zscore_percentage),
                "maximum_abs_zscore": maximum_abs_zscore,
            })

        iqr_outlier_matrix = pd.DataFrame(iqr_masks, index=X_numeric.index)

        if not iqr_outlier_matrix.empty:
            iqr_outlier_count_per_row = iqr_outlier_matrix.sum(axis=1)
            iqr_any_outlier = iqr_outlier_count_per_row > 0
        else:
            iqr_outlier_count_per_row = pd.Series(0, index=X_numeric.index)
            iqr_any_outlier = pd.Series(False, index=X_numeric.index)

        zscore_outlier_matrix = pd.DataFrame(zscore_masks, index=X_numeric.index)

        if not zscore_outlier_matrix.empty:
            zscore_outlier_count_per_row = zscore_outlier_matrix.sum(axis=1)
            zscore_any_outlier = zscore_outlier_count_per_row > 0
        else:
            zscore_outlier_count_per_row = pd.Series(0, index=X_numeric.index)
            zscore_any_outlier = pd.Series(False, index=X_numeric.index)

        imputer = SimpleImputer(strategy="median")
        X_imputed = imputer.fit_transform(X_numeric)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        isolation_result = None
        isolation_labels = pd.Series(
            1, index=X_numeric.index, name="isolation_forest_label"
        )
        isolation_scores = pd.Series(
            np.nan, index=X_numeric.index, name="isolation_forest_score"
        )
        isolation_model = None

        if use_isolation_forest:
            isolation_model = IsolationForest(
                n_estimators=n_estimators,
                contamination=contamination,
                random_state=random_state,
            )
            isolation_labels_array = isolation_model.fit_predict(X_scaled)
            isolation_score_array = isolation_model.decision_function(X_scaled)

            isolation_labels = pd.Series(
                isolation_labels_array,
                index=X_numeric.index,
                name="isolation_forest_label",
            )
            isolation_scores = pd.Series(
                isolation_score_array,
                index=X_numeric.index,
                name="isolation_forest_score",
            )

            isolation_outlier_count = int((isolation_labels == -1).sum())

            isolation_result = {
                "enabled": True,
                "n_estimators": int(n_estimators),
                "contamination": float(contamination),
                "outlier_count": isolation_outlier_count,
                "outlier_pct": float(
                    (isolation_outlier_count / len(isolation_labels)) * 100
                ),
            }

            self.isolation_forest_model = isolation_model
        else:
            isolation_result = {"enabled": False}

        lof_result = None
        lof_labels = pd.Series(1, index=X_numeric.index, name="lof_label")
        lof_scores = pd.Series(
            np.nan, index=X_numeric.index, name="lof_score"
        )
        lof_model = None

        if use_lof:
            effective_neighbors = min(lof_neighbors, len(X_numeric) - 1)

            if effective_neighbors < 1:
                raise ValueError("Not enough observations for LOF.")

            lof_model = LocalOutlierFactor(
                n_neighbors=effective_neighbors,
                contamination=contamination,
            )

            lof_labels_array = lof_model.fit_predict(X_scaled)
            lof_score_array = lof_model.negative_outlier_factor_

            lof_labels = pd.Series(
                lof_labels_array,
                index=X_numeric.index,
                name="lof_label",
            )
            lof_scores = pd.Series(
                lof_score_array,
                index=X_numeric.index,
                name="lof_score",
            )

            lof_outlier_count = int((lof_labels == -1).sum())

            lof_result = {
                "enabled": True,
                "n_neighbors": int(effective_neighbors),
                "contamination": float(contamination),
                "outlier_count": lof_outlier_count,
                "outlier_pct": float(
                    (lof_outlier_count / len(lof_labels)) * 100
                ),
            }

            self.lof_model = lof_model
        else:
            lof_result = {"enabled": False}

        row_results = pd.DataFrame({
            "iqr_outlier_count": iqr_outlier_count_per_row,
            "iqr_any_outlier": iqr_any_outlier,
            "zscore_outlier_count": zscore_outlier_count_per_row,
            "zscore_any_outlier": zscore_any_outlier,
            "isolation_forest_label": isolation_labels,
            "isolation_forest_score": isolation_scores,
            "lof_label": lof_labels,
            "lof_score": lof_scores,
        })

        row_results["methods_flagging_outlier"] = (
            row_results["iqr_any_outlier"].astype(int)
            + row_results["zscore_any_outlier"].astype(int)
            + (
                (row_results["isolation_forest_label"] == -1).astype(int)
                if use_isolation_forest else 0
            )
            + (
                (row_results["lof_label"] == -1).astype(int)
                if use_lof else 0
            )
        )

        row_results["consensus_outlier"] = (
            row_results["methods_flagging_outlier"] >= 2
        )

        total_observations = len(row_results)

        iqr_any_count = int(row_results["iqr_any_outlier"].sum())
        zscore_any_count = int(row_results["zscore_any_outlier"].sum())
        consensus_count = int(row_results["consensus_outlier"].sum())

        overall_summary = {
            "observations": int(total_observations),
            "iqr_any_outlier_count": iqr_any_count,
            "iqr_any_outlier_pct": float(
                (iqr_any_count / total_observations) * 100
            ),
            "zscore_any_outlier_count": zscore_any_count,
            "zscore_any_outlier_pct": float(
                (zscore_any_count / total_observations) * 100
            ),
            "consensus_outlier_count": consensus_count,
            "consensus_outlier_pct": float(
                (consensus_count / total_observations) * 100
            ),
        }

        feature_results_sorted = sorted(
            feature_results,
            key=lambda item: item["iqr_outlier_pct"],
            reverse=True,
        )

        top_feature_results = feature_results_sorted[:top_n_features]

        self.outlier_results_df = row_results

        plot_path = self.plots_dir / filename
        feature_filename = filename.replace(".png", "_features.png")
        feature_plot_path = self.plots_dir / feature_filename
        score_filename = filename.replace(".png", "_scores.png")
        score_plot_path = self.plots_dir / score_filename

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        plot_features = [item["feature"] for item in top_feature_results]
        iqr_counts = [item["iqr_outlier_count"] for item in top_feature_results]

        axes[0, 0].barh(
            plot_features[::-1],
            iqr_counts[::-1],
            color="#4C78A8",
        )
        axes[0, 0].set_title("IQR Outlier Counts")
        axes[0, 0].set_xlabel("Outlier Count")

        z_counts = [item["zscore_outlier_count"] for item in top_feature_results]

        axes[0, 1].barh(
            plot_features[::-1],
            z_counts[::-1],
            color="#F58518",
        )
        axes[0, 1].set_title("Z-Score Outlier Counts")
        axes[0, 1].set_xlabel("Outlier Count")

        method_names = ["IQR", "Z-score"]
        method_counts = [iqr_any_count, zscore_any_count]

        if use_isolation_forest:
            method_names.append("Isolation Forest")
            method_counts.append(int((isolation_labels == -1).sum()))

        if use_lof:
            method_names.append("LOF")
            method_counts.append(int((lof_labels == -1).sum()))

        axes[1, 0].bar(
            method_names,
            method_counts,
            color="#54A24B",
        )
        axes[1, 0].set_title("Outliers by Method")
        axes[1, 0].set_ylabel("Observations")
        axes[1, 0].tick_params(axis="x", rotation=25)

        consensus_values = [
            total_observations - consensus_count,
            consensus_count,
        ]

        axes[1, 1].bar(
            ["No consensus", "Consensus"],
            consensus_values,
            color=["#BAB0AC", "#E45756"],
        )
        axes[1, 1].set_title("Multi-Method Outlier Consensus")
        axes[1, 1].set_ylabel("Observations")

        plt.tight_layout()
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 8))

        feature_plot_data = top_feature_results[::-1]
        feature_names = [item["feature"] for item in feature_plot_data]

        iqr_percentages = [
            item["iqr_outlier_pct"] for item in feature_plot_data
        ]
        z_percentages = [
            item["zscore_outlier_pct"] for item in feature_plot_data
        ]

        y_positions = np.arange(len(feature_names))
        bar_height = 0.35

        ax.barh(
            y_positions - bar_height / 2,
            iqr_percentages,
            height=bar_height,
            label="IQR",
            color="#4C78A8",
        )

        ax.barh(
            y_positions + bar_height / 2,
            z_percentages,
            height=bar_height,
            label="Z-score",
            color="#F58518",
        )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Outlier Percentage (%)")
        ax.set_title("Feature-Level Outlier Rates")
        ax.legend()
        ax.grid(axis="x", alpha=0.25)

        plt.tight_layout()
        fig.savefig(feature_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if use_isolation_forest:
            axes[0].hist(
                isolation_scores.dropna(),
                bins=40,
                color="#4C78A8",
                alpha=0.75,
            )
            axes[0].set_title("Isolation Forest Scores")
            axes[0].set_xlabel("Decision Function Score")
            axes[0].set_ylabel("Frequency")
            axes[0].grid(alpha=0.20)
        else:
            axes[0].text(
                0.5,
                0.5,
                "Isolation Forest disabled",
                ha="center",
                va="center",
                transform=axes[0].transAxes,
            )
            axes[0].set_xticks([])
            axes[0].set_yticks([])

        if use_lof:
            axes[1].hist(
                lof_scores.dropna(),
                bins=40,
                color="#F58518",
                alpha=0.75,
            )
            axes[1].set_title("LOF Scores")
            axes[1].set_xlabel("Negative Outlier Factor")
            axes[1].set_ylabel("Frequency")
            axes[1].grid(alpha=0.20)
        else:
            axes[1].text(
                0.5,
                0.5,
                "LOF disabled",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
            )
            axes[1].set_xticks([])
            axes[1].set_yticks([])

        plt.tight_layout()
        fig.savefig(score_plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        result = {
            "component": component_name,
            "status": "success",
            "features_used": [str(feature) for feature in X_numeric.columns],
            "feature_count": int(X_numeric.shape[1]),
            "observations_used": int(len(X_numeric)),
            "parameters": {
                "contamination": float(contamination),
                "z_threshold": float(z_threshold),
                "iqr_multiplier": float(iqr_multiplier),
                "use_isolation_forest": bool(use_isolation_forest),
                "use_lof": bool(use_lof),
                "lof_neighbors": int(lof_neighbors),
                "n_estimators": int(n_estimators),
                "random_state": (
                    int(random_state)
                    if random_state is not None
                    else None
                ),
            },
            "overall": overall_summary,
            "feature_analysis": feature_results,
            "top_outlier_features": top_feature_results,
            "isolation_forest": isolation_result,
            "local_outlier_factor": lof_result,
            "plots": {
                "main": str(plot_path),
                "features": str(feature_plot_path),
                "scores": str(score_plot_path),
            },
        }

        self.outlier_results = result
        self.results[component_name] = result
        self.component_status[component_name] = {
            "status": "success",
            "main_plot": str(plot_path),
            "feature_plot": str(feature_plot_path),
            "score_plot": str(score_plot_path),
        }

        return result

    except Exception as exc:
        error_result = {
            "component": component_name,
            "status": "error",
            "target": getattr(self, "target", None),
            "target_type": getattr(self, "target_type", None),
            "error": str(exc),
        }

        self.results[component_name] = error_result
        self.component_status[component_name] = {
            "status": "error",
            "error": str(exc),
        }

        return error_result


# ========================================================
# VISUALIZATIONS DASHBOARD
# ========================================================

def visualizations(
    self,
    filename: str = "eda_overview.png",
    dpi: int = 150,
    figsize: tuple[int, int] = (18, 12),
) -> dict[str, Any]:
    """Collect existing component plots and create an EDA dashboard."""


    component_name = "visualizations"

    component_order = [
        "target_distribution", "missingness", "pearson_spearman",
        "categorical_analysis", "preprocessing", "mutual_information",
        "random_forest", "permutation_importance", "cross_validation",
        "logistic_regression", "decision_tree", "iv_woe",
        "kmeans", "pca", "outliers",
    ]

    plot_keys = [
        "plot", "main_plot", "importance_plot", "variance_plot",
        "scores_plot", "feature_plot", "features_plot", "roc_plot",
        "confusion_matrix_plot", "cv_plot", "diagnostic_plot",
        "diagnostics_plot", "plots",
    ]

    try:
        # Validation
        if dpi <= 0:
            raise ValueError("dpi must be greater than 0.")
        if not isinstance(figsize, tuple) or len(figsize) != 2:
            raise ValueError(
                "figsize must be a tuple containing (width, height)."
            )

        self.plots_dir.mkdir(parents=True, exist_ok=True)

        component_status = getattr(self, "component_status", {})
        plot_map = {}

        # ---------------------------------------------------------
        # Collect plots registered in component_status
        # ---------------------------------------------------------
        for component in component_order:
            status = component_status.get(component, {})
            if not isinstance(status, dict):
                continue

            paths = []
            for key in plot_keys:
                value = status.get(key)

                if isinstance(value, str):
                    paths.append(value)

                elif isinstance(value, list):
                    paths.extend(
                        x for x in value
                        if isinstance(x, str)
                    )

                elif isinstance(value, dict):
                    paths.extend(
                        x for x in value.values()
                        if isinstance(x, str)
                    )

            if paths:
                plot_map[component] = list(dict.fromkeys(paths))

        # ---------------------------------------------------------
        # Discover unregistered PNG files
        # ---------------------------------------------------------
        registered = {
            str(Path(path))
            for paths in plot_map.values()
            for path in paths
        }

        for png in self.plots_dir.glob("*.png"):
            path = str(png)
            if path in registered:
                continue

            name = png.name.lower()
            component = next(
                (
                    c for c in component_order
                    if c.lower() in name
                ),
                "other",
            )
            plot_map.setdefault(component, []).append(path)

        # ---------------------------------------------------------
        # Resolve valid paths
        # ---------------------------------------------------------
        valid_plots = {}

        for component, paths in plot_map.items():
            valid = []

            for path in paths:
                p = Path(path)

                if not p.is_absolute():
                    p = p if p.exists() else self.plots_dir / p.name

                if p.exists():
                    valid.append(str(p))

            if valid:
                valid_plots[component] = valid

        dashboard_path = self.plots_dir / filename

        # ---------------------------------------------------------
        # Select one representative plot per component
        # ---------------------------------------------------------
        dashboard_images = []

        for component in component_order:
            paths = valid_plots.get(component, [])
            if paths:
                p = Path(paths[0])
                if p.resolve() != dashboard_path.resolve():
                    dashboard_images.append((component, paths[0]))

        for component, paths in valid_plots.items():
            if component in component_order:
                continue

            for path in paths:
                if Path(path).resolve() != dashboard_path.resolve():
                    dashboard_images.append((component, path))

        dashboard_images = dashboard_images[:12]

        # ---------------------------------------------------------
        # Create dashboard
        # ---------------------------------------------------------
        if dashboard_images:
            n_cols = 3
            n_rows = int(np.ceil(len(dashboard_images) / n_cols))

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(figsize[0], max(figsize[1], 4 * n_rows)),
                squeeze=False,
            )

            axes = axes.flatten()

            for ax, (component, path) in zip(
                axes, dashboard_images
            ):
                try:
                    ax.imshow(Image.open(path))
                    ax.set_title(
                        component.replace("_", " ").title(),
                        fontsize=11,
                        fontweight="bold",
                    )
                except Exception:
                    ax.text(
                        0.5, 0.5,
                        f"Unable to load\n{component}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                ax.axis("off")

            for ax in axes[len(dashboard_images):]:
                ax.axis("off")

            fig.suptitle(
                "TargetEDA Visualization Overview",
                fontsize=18,
                fontweight="bold",
                y=0.995,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            fig.savefig(
                dashboard_path,
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)

            dashboard_created = True

        else:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(
                0.5,
                0.5,
                "No component visualizations available",
                ha="center",
                va="center",
                fontsize=16,
            )
            ax.axis("off")
            fig.savefig(
                dashboard_path,
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close(fig)

            dashboard_created = False

        # ---------------------------------------------------------
        # Component status summary
        # ---------------------------------------------------------
        successful = []
        failed = []
        skipped = []

        for component in component_order:
            status = component_status.get(component, {})
            if not isinstance(status, dict):
                continue

            value = status.get("status")

            if value == "success":
                successful.append(component)
            elif value == "error":
                failed.append(component)
            elif value == "skipped":
                skipped.append(component)

        total_plots = sum(
            len(paths)
            for paths in valid_plots.values()
        )

        result = {
            "component": component_name,
            "status": "success",
            "dashboard": (
                str(dashboard_path)
                if dashboard_created else None
            ),
            "dashboard_created": bool(dashboard_created),
            "individual_plot_count": int(total_plots),
            "dashboard_plot_count": int(len(dashboard_images)),
            "successful_components": successful,
            "failed_components": failed,
            "skipped_components": skipped,
            "component_plots": valid_plots,
        }

        self.visualization_results = result
        self.results[component_name] = result
        self.component_status[component_name] = {
            "status": "success",
            "dashboard": (
                str(dashboard_path)
                if dashboard_created else None
            ),
            "individual_plot_count": int(total_plots),
            "dashboard_plot_count": int(len(dashboard_images)),
        }

        return result

    except Exception as exc:
        error_result = {
            "component": component_name,
            "status": "error",
            "target": getattr(self, "target", None),
            "target_type": getattr(self, "target_type", None),
            "error": str(exc),
        }

        self.results[component_name] = error_result
        self.component_status[component_name] = {
            "status": "error",
            "error": str(exc),
        }

        return error_result


# ========================================================
# FINAL ANALYSIS PIPELINE
# ========================================================


def analyze_target(
    self,
    run_all: bool = True,
    components: list[str] | None = None,
    continue_on_error: bool = True,
    save_json: bool = True,
    json_filename: str = "target_analysis.json",
    visualization: bool = True,
) -> dict[str, Any]:
    """Run the complete TargetEDA analytical pipeline."""

    import json
    import time

    component_name = "analyze_target"
    start = time.perf_counter()

    if not hasattr(self, "results"):
        self.results = {}
    if not hasattr(self, "component_status"):
        self.component_status = {}

    component_order = [
        "target_distribution",
        "missingness",
        "pearson_spearman",
        "categorical_analysis",
        "preprocessing",
        "mutual_information",
        "random_forest",
        "permutation_importance",
        "cross_validation",
        "logistic_regression",
        "decision_tree",
        "iv_woe",
        "kmeans",
        "pca",
        "outliers",
    ]

    component_methods = {
        component: component
        for component in component_order
    }

    # ---------------------------------------------------------
    # Select components
    # ---------------------------------------------------------
    if components is None:
        selected = component_order.copy() if run_all else []
    else:
        selected = []
        for component in components:
            if component not in component_order:
                raise ValueError(
                    f"Unknown component: {component}. "
                    f"Valid components are: {component_order}"
                )
            if component not in selected:
                selected.append(component)

    execution = {
        "requested": selected,
        "successful": [],
        "failed": [],
        "skipped": [],
    }

    metadata = {
        "target": getattr(self, "target", None),
        "target_type": getattr(self, "target_type", None),
        "rows": int(len(self.df)) if hasattr(self, "df") else None,
        "columns": int(len(self.df.columns)) if hasattr(self, "df") else None,
    }

    # ---------------------------------------------------------
    # Run components
    # ---------------------------------------------------------
    for component in selected:
        method_name = component_methods[component]
        method = getattr(self, method_name, None)

        if method is None:
            reason = f"Method '{method_name}' is not implemented."
            result = {
                "component": component,
                "status": "skipped",
                "reason": reason,
            }

            self.results[component] = result
            self.component_status[component] = {
                "status": "skipped",
                "reason": reason,
            }
            execution["skipped"].append(component)
            continue

        component_start = time.perf_counter()

        try:
            result = method()

            if result is None:
                result = {
                    "component": component,
                    "status": "success",
                    "message": (
                        "Component completed without returning "
                        "a result dictionary."
                    ),
                }

            status = (
                result.get("status", "success")
                if isinstance(result, dict)
                else "success"
            )

            elapsed = time.perf_counter() - component_start

            if isinstance(result, dict):
                result["execution_time_seconds"] = float(elapsed)

            self.results[component] = result

            if status == "success":
                execution["successful"].append(component)
                self.component_status[component] = {
                    "status": "success",
                    "execution_time_seconds": float(elapsed),
                }

            elif status == "skipped":
                execution["skipped"].append(component)
                self.component_status[component] = {
                    "status": "skipped",
                    "execution_time_seconds": float(elapsed),
                }

            else:
                execution["failed"].append(component)
                self.component_status[component] = {
                    "status": "error",
                    "execution_time_seconds": float(elapsed),
                }

                if not continue_on_error:
                    raise RuntimeError(
                        f"Component '{component}' returned an error."
                    )

        except Exception as exc:
            elapsed = time.perf_counter() - component_start

            result = {
                "component": component,
                "status": "error",
                "error": str(exc),
                "execution_time_seconds": float(elapsed),
            }

            self.results[component] = result
            self.component_status[component] = {
                "status": "error",
                "error": str(exc),
                "execution_time_seconds": float(elapsed),
            }
            execution["failed"].append(component)

            if not continue_on_error:
                raise

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------
    if visualization:
        visualize = getattr(self, "visualizations", None)

        if visualize is not None:
            viz_start = time.perf_counter()

            try:
                visualization_result = visualize()
                elapsed = time.perf_counter() - viz_start

                if isinstance(visualization_result, dict):
                    visualization_result["execution_time_seconds"] = float(
                        elapsed
                    )

            except Exception as exc:
                visualization_result = {
                    "component": "visualizations",
                    "status": "error",
                    "error": str(exc),
                }

        else:
            visualization_result = {
                "component": "visualizations",
                "status": "skipped",
                "reason": "visualizations() is not implemented.",
            }

        self.results["visualizations"] = visualization_result

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------
    total_time = time.perf_counter() - start

    final_result = {
        "analysis": {
            "name": "TargetEDA",
            "version": getattr(self, "version", "1.0.0"),
            "status": (
                "success"
                if not execution["failed"]
                else "completed_with_errors"
            ),
            "execution_time_seconds": float(total_time),
        },
        "metadata": metadata,
        "execution": execution,
        "results": self.results,
    }

    # ---------------------------------------------------------
    # JSON-safe conversion
    # ---------------------------------------------------------
    try:
        final_result = make_json_safe(final_result)
    except Exception:
        final_result = json.loads(
            json.dumps(final_result, default=str)
        )

    # ---------------------------------------------------------
    # Save JSON
    # ---------------------------------------------------------
    if save_json:
        if hasattr(self, "output_dir"):
            output_dir = Path(self.output_dir)
        else:
            output_dir = Path(
                getattr(
                    self,
                    "plots_dir",
                    Path("target_analysis"),
                )
            ).parent

        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / json_filename

        with open(json_path, "w", encoding="utf-8") as file:
            json.dump(
                final_result,
                file,
                indent=2,
                ensure_ascii=False,
            )

        final_result["analysis"]["json_path"] = str(json_path)

    # ---------------------------------------------------------
    # Store final result
    # ---------------------------------------------------------
    self.analysis_result = final_result

    self.results[component_name] = {
        "status": final_result["analysis"]["status"],
        "execution_time_seconds": (
            final_result["analysis"]["execution_time_seconds"]
        ),
    }

    return final_result
