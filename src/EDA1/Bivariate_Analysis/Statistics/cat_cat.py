import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
from src.config import BASE_DIR, config

# ============================================================
# Configuration & Utilities
# ============================================================

OUTPUT_DIR = BASE_DIR / "src" / "EDA1" / "Bivariate_Analysis" / "Reports"
WORKBOOK_NAME = "cat_cat_eda.xlsx"
LOG_FILE_NAME = "cat_cat_eda.log"
columns_json_path = BASE_DIR / "src" / "Data" / "classified_columns.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(
            OUTPUT_DIR / LOG_FILE_NAME, mode="a", encoding="utf-8"
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_categorical_columns() -> list[str]:
    with open(columns_json_path, "r") as file:
        columns_dict = json.load(file)
    return list(columns_dict["bivariate_candidates"]["categorical"]) 


def get_valid_pair(df: pd.DataFrame, f1: str, f2: str) -> pd.DataFrame:
    return df[[f1, f2]].dropna()


def get_contingency_table(df: pd.DataFrame, f1: str, f2: str) -> pd.DataFrame:
    pair = get_valid_pair(df, f1, f2)
    return pd.crosstab(pair[f1], pair[f2], dropna=False)


# ============================================================
# 1. Unified Contingency Summary (Counts & Percentages)
# ============================================================


def generate_contingency_summary(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = get_categorical_columns()
    logger.info(
        f"Computing unified contingency summary for {len(cat_cols)} categorical features..."
    )
    results = []

    for i, f1 in enumerate(cat_cols):
        for f2 in cat_cols[i + 1:]:
            try:
                table = get_contingency_table(df, f1, f2)
                total = table.to_numpy().sum()
                if total == 0:
                    continue
                row_totals = table.sum(axis=1)
                col_totals = table.sum(axis=0)

                for c1 in table.index:
                    for c2 in table.columns:
                        count = int(table.loc[c1, c2])
                        row_tot = int(row_totals.loc[c1])
                        col_tot = int(col_totals.loc[c2])

                        overall_pct = (count / total) * 100 if total > 0 else 0.0
                        row_pct = (count / row_tot) * 100 if row_tot > 0 else 0.0
                        col_pct = (count / col_tot) * 100 if col_tot > 0 else 0.0

                        results.append(
                            {
                                "feature_1": f1,
                                "feature_2": f2,
                                "category_1": str(c1),
                                "category_2": str(c2),
                                "count": count,
                                "overall_percentage": overall_pct,
                                "row_total": row_tot,
                                "row_percentage": row_pct,
                                "column_total": col_tot,
                                "column_percentage": col_pct,
                            }
                        )
                logger.info(
                    f"Appended contingency summary results for pair ({f1}, {f2})"
                )
            except Exception as e:
                logger.warning(
                    f"Error computing contingency summary for ({f1}, {f2}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# 2. Chi-Square Test
# ============================================================


def generate_chi_square(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = get_categorical_columns()
    logger.info("Computing Chi-Square tests...")
    results = []

    for i, f1 in enumerate(cat_cols):
        for f2 in cat_cols[i + 1:]:
            try:
                table = get_contingency_table(df, f1, f2)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                chi2, p_val, dof, expected = chi2_contingency(table)
                results.append(
                    {
                        "feature_1": f1,
                        "feature_2": f2,
                        "sample_size": int(table.to_numpy().sum()),
                        "rows": table.shape[0],
                        "columns": table.shape[1],
                        "chi_square_statistic": chi2,
                        "degrees_of_freedom": dof,
                        "p_value": p_val,
                        "min_expected_count": expected.min(),
                        "expected_count_below_5": int((expected < 5).sum()),
                    }
                )
                logger.info(
                    f"Appended Chi-Square for ({f1}, {f2}) | chi2: {chi2:.4f}, p: {p_val:.4e}"
                )
            except Exception as e:
                logger.warning(f"Error computing Chi-Square for ({f1}, {f2}): {e}")

    return pd.DataFrame(results)


# ============================================================
# 3. Cramer's V
# ============================================================


def generate_cramers_v(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = get_categorical_columns()
    logger.info("Computing Cramer's V...")
    results = []

    for i, f1 in enumerate(cat_cols):
        for f2 in cat_cols[i + 1:]:
            try:
                table = get_contingency_table(df, f1, f2)
                if table.shape[0] < 2 or table.shape[1] < 2:
                    continue

                chi2, p_val, dof, _ = chi2_contingency(table)
                n = table.to_numpy().sum()
                if n <= 0:
                    continue

                min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
                if min_dim <= 0:
                    continue

                cramers_v = np.sqrt((chi2 / n) / min_dim)
                results.append(
                    {
                        "feature_1": f1,
                        "feature_2": f2,
                        "sample_size": int(n),
                        "chi_square_statistic": chi2,
                        "degrees_of_freedom": dof,
                        "p_value": p_val,
                        "cramers_v": cramers_v,
                    }
                )
                logger.info(
                    f"Appended Cramer's V for ({f1}, {f2}) | V: {cramers_v:.4f}"
                )
            except Exception as e:
                logger.warning(f"Error computing Cramer's V for ({f1}, {f2}): {e}")

    return pd.DataFrame(results)


# ============================================================
# 4. Fisher's Exact Test
# ============================================================


def generate_fisher_exact(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = get_categorical_columns()
    logger.info("Computing Fisher's Exact tests...")
    results = []

    for i, f1 in enumerate(cat_cols):
        for f2 in cat_cols[i + 1:]:
            try:
                table = get_contingency_table(df, f1, f2)
                if table.shape != (2, 2):
                    continue

                odds_ratio, p_val = fisher_exact(table.to_numpy())
                results.append(
                    {
                        "feature_1": f1,
                        "feature_2": f2,
                        "category_1_level_1": str(table.index[0]),
                        "category_1_level_2": str(table.index[1]),
                        "category_2_level_1": str(table.columns[0]),
                        "category_2_level_2": str(table.columns[1]),
                        "n_11": int(table.iloc[0, 0]),
                        "n_12": int(table.iloc[0, 1]),
                        "n_21": int(table.iloc[1, 0]),
                        "n_22": int(table.iloc[1, 1]),
                        "odds_ratio": odds_ratio,
                        "p_value": p_val,
                    }
                )
                logger.info(
                    f"Appended Fisher's Exact for ({f1}, {f2}) | OR: {odds_ratio:.4f}, p: {p_val:.4e}"
                )
            except Exception as e:
                logger.warning(
                    f"Error computing Fisher's Exact test for ({f1}, {f2}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# Workbook Writer, Runner & Entry Point
# ============================================================


def save_workbook(results: dict[str, pd.DataFrame]) -> None:
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        wb_path = OUTPUT_DIR / WORKBOOK_NAME
        with pd.ExcelWriter(wb_path, engine="openpyxl") as writer:
            for sheet, df in results.items():
                df.to_excel(writer, sheet_name=sheet, index=False)
        logger.info(f"Workbook successfully created at: {wb_path}")
    except Exception as e:
        logger.error(f"Failed to save Excel workbook: {e}")
        raise


def run_cat_cat_eda(df: pd.DataFrame) -> None:
    logger.info("Starting CAT-CAT exploratory data analysis...")
    results = {
        "contingency_summary": generate_contingency_summary(df),
        "chi_square": generate_chi_square(df),
        "cramers_v": generate_cramers_v(df),
        "fisher_exact": generate_fisher_exact(df),
    }
    save_workbook(results)
    logger.info("CAT-CAT EDA pipeline completed successfully.")


def cat_cat_eda(file_path: Path) -> None:
    logger.info(f"Attempting to load dataset from: {file_path}")
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        df = pd.read_csv(file_path)
        logger.info(f"Dataset successfully loaded with shape: {df.shape}")
        run_cat_cat_eda(df)
    except (FileNotFoundError, pd.errors.EmptyDataError) as err:
        logger.error(err)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")
        raise


if __name__ == "__main__":
    cat_cat_eda(file_path=BASE_DIR/ "src"/ "Data"/ "cleaned_ecommerce_dataset.csv")