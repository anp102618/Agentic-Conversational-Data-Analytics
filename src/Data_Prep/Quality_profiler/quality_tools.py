from __future__ import annotations

import re
import numpy as np
import pandas as pd


class DeterministicProfiler:
    """Creates compact, deterministic evidence for each column and dataset."""

    NEAR_CONSTANT_RATIO = 0.95
    HIGH_CARDINALITY_RATIO = 0.50
    HIGH_CARDINALITY_MIN_UNIQUE = 20
    ID_UNIQUENESS_RATIO = 0.95
    CLASS_IMBALANCE_RATIO = 0.80
    OUTLIER_IQR_MULTIPLIER = 1.5

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        categorical_allowed_values: dict[str, set[str]] | None = None,
    ):
        self.df = df
        self.target_column = target_column
        self.categorical_allowed_values = categorical_allowed_values or {}

    def dataset_info(self) -> dict:
        rows, columns = self.df.shape
        return {"rows": int(rows), "columns": int(columns), "cells": int(rows * columns)}

    def column_profiles(self) -> list[dict]:
        return [self.profile_column(column) for column in self.df.columns]

    def profile_column(self, column: str) -> dict:
        series = self.df[column]
        row_count = len(series)
        non_null_count = int(series.notna().sum())
        missing_count = int(series.isna().sum())
        semantic_type = self._infer_semantic_type(series)

        profile = {
            "column": column,
            "physical_dtype": str(series.dtype),
            "row_count": row_count,
            "non_null_count": non_null_count,
            "missing_count": missing_count,
            "missing_percentage": round(missing_count / row_count * 100, 2) if row_count else 0.0,
            "semantic_type": semantic_type,
            "unique_count": int(series.nunique(dropna=True)),
        }

        profile.update(
            self._numeric_profile(series)
            if pd.api.types.is_numeric_dtype(series)
            else self._categorical_profile(series)
        )
        profile["issues"] = self._column_issues(column, series, semantic_type)
        return profile

    def _numeric_profile(self, series: pd.Series) -> dict:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if numeric.empty:
            return {}

        q1, q3 = float(numeric.quantile(0.25)), float(numeric.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - self.OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + self.OUTLIER_IQR_MULTIPLIER * iqr
        outliers = (numeric < lower) | (numeric > upper)

        return {
            "min": float(numeric.min()),
            "max": float(numeric.max()),
            "mean": round(float(numeric.mean()), 4),
            "median": round(float(numeric.median()), 4),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "outlier_count": int(outliers.sum()),
            "outlier_percentage": round(float(outliers.mean() * 100), 2),
        }

    def _categorical_profile(self, series: pd.Series) -> dict:
        values = series.dropna().astype(str)
        counts = values.value_counts().head(15).to_dict()

        return {
            "top_values": {str(k): int(v) for k, v in counts.items()},
            "has_leading_spaces": bool(values.str.match(r"^\s").any()),
            "has_trailing_spaces": bool(values.str.match(r"\s$").any()),
            "has_multiple_spaces": bool(values.str.contains(r"\s{2,}", regex=True).any()),
        }

    def _infer_semantic_type(self, series: pd.Series) -> str:
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        if pd.api.types.is_numeric_dtype(series):
            return "numeric"

        values = series.dropna().astype(str)
        if values.empty:
            return "unknown"

        datetime_values = pd.to_datetime(values, errors="coerce")
        if datetime_values.notna().mean() >= 0.95:
            return "datetime_string"

        numeric_values = pd.to_numeric(values, errors="coerce")
        if numeric_values.notna().mean() >= 0.95:
            return "numeric_string"

        unique_ratio = values.nunique() / max(len(values), 1)
        return "categorical" if unique_ratio < 0.05 else "string"

    def duplicate_count(self) -> int:
        return int(self.df.duplicated().sum())

    def missing_value_evidence(self) -> dict:
        total_missing = int(self.df.isna().sum().sum())
        columns_with_missing = {}

        for column in self.df.columns:
            count = int(self.df[column].isna().sum())
            if count:
                columns_with_missing[column] = {
                    "count": count,
                    "percentage": round(count / max(len(self.df), 1) * 100, 2),
                }

        return {
            "total_missing_cells": total_missing,
            "columns_with_missing": columns_with_missing,
        }

    def _constant_issue(self, series: pd.Series) -> str | None:
        non_null = series.dropna()
        if non_null.empty:
            return None

        if non_null.nunique() == 1:
            return "constant_column"

        if non_null.value_counts().iloc[0] / len(non_null) >= self.NEAR_CONSTANT_RATIO:
            return "near_constant_column"

        return None

    def _is_high_cardinality(self, series: pd.Series) -> bool:
        non_null = series.dropna()
        if len(non_null) == 0:
            return False

        unique_count = int(non_null.nunique())
        return (
            unique_count >= self.HIGH_CARDINALITY_MIN_UNIQUE
            and unique_count / len(non_null) >= self.HIGH_CARDINALITY_RATIO
        )

    def _is_id_column(self, column: str, series: pd.Series) -> bool:
        pattern = re.compile(
            r"(^id$|_id$|^id_|identifier|customerid|userid|accountid|recordid|uuid)"
        )

        if pattern.search(str(column).lower()):
            return True

        non_null = series.dropna()
        if non_null.empty:
            return False

        return (
            non_null.nunique() / len(non_null) >= self.ID_UNIQUENESS_RATIO
            and len(non_null) >= 10
        )

    def _invalid_value_evidence(self, series: pd.Series, semantic_type: str) -> dict:
        invalid_count = 0
        examples = []

        if semantic_type == "numeric_string":
            converted = pd.to_numeric(series, errors="coerce")
            invalid_mask = series.notna() & converted.isna()
        elif semantic_type in {"datetime", "datetime_string"}:
            converted = pd.to_datetime(series, errors="coerce")
            invalid_mask = series.notna() & converted.isna()
        else:
            return {"count": 0, "examples": []}

        invalid_count = int(invalid_mask.sum())
        examples = series[invalid_mask].astype(str).head(10).tolist()
        return {"count": invalid_count, "examples": examples}

    def _invalid_categorical_values(self, column: str, series: pd.Series) -> dict:
        allowed = self.categorical_allowed_values.get(column)
        if not allowed:
            return {"count": 0, "values": []}

        values = series.dropna().astype(str)
        invalid = values[~values.isin(allowed)]

        return {
            "count": int(len(invalid)),
            "values": invalid.value_counts().head(10).to_dict(),
        }

    def _datetime_issues(self, series: pd.Series, semantic_type: str) -> list[str]:
        if semantic_type not in {"datetime", "datetime_string"}:
            return []

        converted = pd.to_datetime(series, errors="coerce")
        valid = converted.dropna()
        if valid.empty:
            return []

        issues = []

        if (valid > pd.Timestamp.now()).sum() > 0:
            issues.append("future_datetime_values")

        if (valid < pd.Timestamp("1900-01-01")).sum() > 0:
            issues.append("suspicious_old_datetime_values")

        try:
            timezone_count = len({getattr(value, "tzinfo", None) for value in valid})
            if timezone_count > 1:
                issues.append("mixed_timezone_values")
        except Exception:
            pass

        return issues

    def _class_imbalance(self, column: str, series: pd.Series) -> dict | None:
        if self.target_column != column:
            return None

        values = series.dropna()
        if values.empty:
            return None

        counts = values.value_counts()
        if len(counts) < 2:
            return None

        majority_ratio = counts.iloc[0] / len(values)
        minority_ratio = counts.iloc[-1] / len(values)

        return {
            "is_imbalanced": bool(majority_ratio >= self.CLASS_IMBALANCE_RATIO),
            "majority_class": str(counts.index[0]),
            "majority_percentage": round(majority_ratio * 100, 2),
            "minority_class": str(counts.index[-1]),
            "minority_percentage": round(minority_ratio * 100, 2),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
        }

    def _potential_leakage(self, column: str, series: pd.Series) -> list[str]:
        if self.target_column is None or column == self.target_column:
            return []

        issues = []
        column_name = str(column).lower()

        leakage_keywords = [
            "target", "label", "outcome", "result",
            "prediction", "predicted", "future", "post_", "after_",
        ]

        if any(keyword in column_name for keyword in leakage_keywords):
            issues.append("suspicious_column_name")

        target = self.df[self.target_column]

        if series.notna().sum() > 0 and target.notna().sum() > 0:
            aligned = pd.DataFrame({"feature": series, "target": target}).dropna()

            if not aligned.empty:
                try:
                    if (aligned["feature"] == aligned["target"]).mean() >= 0.95:
                        issues.append("feature_nearly_equals_target")
                except Exception:
                    pass

                try:
                    feature_strings = aligned["feature"].astype(str)
                    target_strings = aligned["target"].astype(str)
                    contains_ratio = feature_strings.str.contains(
                        target_strings, regex=False
                    ).mean()

                    if contains_ratio >= 0.95:
                        issues.append("feature_contains_target")
                except Exception:
                    pass

        return issues

    def _column_issues(
        self,
        column: str,
        series: pd.Series,
        semantic_type: str,
    ) -> dict:
        issues = {}

        missing_count = int(series.isna().sum())
        if missing_count:
            issues["missing_values"] = {
                "count": missing_count,
                "percentage": round(
                    missing_count / max(len(series), 1) * 100, 2
                ),
            }

        constant_issue = self._constant_issue(series)
        if constant_issue:
            issues[constant_issue] = True

        if self._is_high_cardinality(series):
            issues["high_cardinality"] = True

        if self._is_id_column(column, series):
            issues["id_column"] = True

        invalid_values = self._invalid_value_evidence(series, semantic_type)
        if invalid_values["count"] > 0:
            issues["invalid_values"] = invalid_values

        invalid_categories = self._invalid_categorical_values(column, series)
        if invalid_categories["count"] > 0:
            issues["invalid_categorical_values"] = invalid_categories

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").dropna()

            if not numeric.empty:
                q1 = numeric.quantile(0.25)
                q3 = numeric.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.OUTLIER_IQR_MULTIPLIER * iqr
                upper = q3 + self.OUTLIER_IQR_MULTIPLIER * iqr
                mask = (numeric < lower) | (numeric > upper)
                count = int(mask.sum())

                if count:
                    issues["outliers"] = {
                        "count": count,
                        "percentage": round(count / len(numeric) * 100, 2),
                        "lower_bound": float(lower),
                        "upper_bound": float(upper),
                    }

        datetime_issues = self._datetime_issues(series, semantic_type)
        if datetime_issues:
            issues["datetime_issues"] = datetime_issues

        class_imbalance = self._class_imbalance(column, series)
        if class_imbalance and class_imbalance["is_imbalanced"]:
            issues["class_imbalance"] = class_imbalance

        leakage = self._potential_leakage(column, series)
        if leakage:
            issues["potential_leakage"] = leakage

        return issues

    def column_evidence(self, column: str) -> dict:
        return self.profile_column(column)

    def build_evidence(self) -> dict:
        column_profiles = self.column_profiles()
        issue_summary = {
            profile["column"]: profile["issues"]
            for profile in column_profiles
            if profile.get("issues")
        }

        return {
            "dataset": self.dataset_info(),
            "columns": column_profiles,
            "duplicate_rows": self.duplicate_count(),
            "missing_values": self.missing_value_evidence(),
            "issue_summary": issue_summary,
        }
