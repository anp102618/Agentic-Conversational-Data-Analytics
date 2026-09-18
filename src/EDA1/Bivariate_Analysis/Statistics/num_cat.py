from itertools import combinations
import logging
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import f, kruskal, mannwhitneyu, ttest_ind
from src.config import BASE_DIR, config

# ============================================================
# Configuration & Utilities
# ============================================================

OUTPUT_DIR = BASE_DIR / "src" / "EDA1" / "Bivariate_Analysis" / "Reports"
WORKBOOK_NAME = "num_cat_eda.xlsx"
LOG_FILE_NAME = "num_cat_eda.log"
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


def get_numeric_columns() -> list[str]:
    with open(columns_json_path, "r") as file:
        columns_dict = json.load(file)
    return list(columns_dict["bivariate_candidates"]["numeric"]) 

def get_categorical_columns() -> list[str]:
    with open(columns_json_path, "r") as file:
        columns_dict = json.load(file)
    return list(columns_dict["bivariate_candidates"]["categorical"]) 


def get_valid_pair(
    df: pd.DataFrame, num_feat: str, cat_feat: str
) -> pd.DataFrame:
    return df[[num_feat, cat_feat]].dropna()


def get_groups(
    df: pd.DataFrame, num_feat: str, cat_feat: str
) -> dict[str, np.ndarray]:
    pair = get_valid_pair(df, num_feat, cat_feat)
    return {
        str(cat): grp[num_feat].to_numpy()
        for cat, grp in pair.groupby(cat_feat, observed=True)
        if len(grp[num_feat]) > 0
    }


# ============================================================
# 1. Group Statistics
# ============================================================


def generate_group_statistics(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = get_numeric_columns(), get_categorical_columns()
    logger.info(
        f"Computing group statistics for {len(num_cols)} numeric and {len(cat_cols)} categorical features..."
    )
    results = []

    for nf in num_cols:
        for cf in cat_cols:
            try:
                pair = get_valid_pair(df, nf, cf)
                for cat, group in pair.groupby(cf, observed=True):
                    vals = group[nf].dropna()
                    if vals.empty:
                        continue
                    q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
                    results.append(
                        {
                            "numeric_feature": nf,
                            "categorical_feature": cf,
                            "category": str(cat),
                            "sample_size": len(vals),
                            "mean": vals.mean(),
                            "median": vals.median(),
                            "std": vals.std(),
                            "variance": vals.var(),
                            "min": vals.min(),
                            "q25": q25,
                            "q75": q75,
                            "max": vals.max(),
                            "iqr": q75 - q25,
                        }
                    )
                    logger.info(
                        f"Appended group statistics for ({nf}, {cf}), category: '{cat}' | n: {len(vals)}, mean: {vals.mean():.4f}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error computing group statistics for ({nf}, {cf}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# 2. Welch's t-test
# ============================================================


def generate_welch_t_test(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = get_numeric_columns(), get_categorical_columns()
    logger.info("Computing Welch's t-tests...")
    results = []

    for nf in num_cols:
        for cf in cat_cols:
            try:
                groups = get_groups(df, nf, cf)
                if len(groups) != 2:
                    continue
                g_names = list(groups.keys())
                g1, g2 = groups[g_names[0]], groups[g_names[1]]

                if (
                    len(g1) < 2
                    or len(g2) < 2
                    or (np.std(g1, ddof=1) == 0 and np.std(g2, ddof=1) == 0)
                ):
                    continue

                stat, p_val = ttest_ind(g1, g2, equal_var=False)
                results.append(
                    {
                        "numeric_feature": nf,
                        "categorical_feature": cf,
                        "group_1": g_names[0],
                        "group_2": g_names[1],
                        "n_group_1": len(g1),
                        "n_group_2": len(g2),
                        "mean_group_1": np.mean(g1),
                        "mean_group_2": np.mean(g2),
                        "welch_t_statistic": stat,
                        "p_value": p_val,
                    }
                )
                logger.info(
                    f"Appended Welch t-test for ({nf}, {cf}) | t: {stat:.4f}, p: {p_val:.4e}"
                )
            except Exception as e:
                logger.warning(
                    f"Error computing Welch t-test for ({nf}, {cf}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# 3. Mann-Whitney U Test
# ============================================================


def generate_mann_whitney_u(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = get_numeric_columns(), get_categorical_columns()
    logger.info("Computing Mann-Whitney U tests...")
    results = []

    for nf in num_cols:
        for cf in cat_cols:
            try:
                groups = get_groups(df, nf, cf)
                if len(groups) != 2:
                    continue
                g_names = list(groups.keys())
                g1, g2 = groups[g_names[0]], groups[g_names[1]]
                if len(g1) == 0 or len(g2) == 0:
                    continue

                stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
                results.append(
                    {
                        "numeric_feature": nf,
                        "categorical_feature": cf,
                        "group_1": g_names[0],
                        "group_2": g_names[1],
                        "n_group_1": len(g1),
                        "n_group_2": len(g2),
                        "median_group_1": np.median(g1),
                        "median_group_2": np.median(g2),
                        "u_statistic": stat,
                        "p_value": p_val,
                    }
                )
                logger.info(
                    f"Appended Mann-Whitney U for ({nf}, {cf}) | U: {stat:.4f}, p: {p_val:.4e}"
                )
            except Exception as e:
                logger.warning(
                    f"Error computing Mann-Whitney U for ({nf}, {cf}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# Welch ANOVA Helper & Function
# ============================================================


def _welch_anova(groups: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    valid = [v for v in groups.values() if len(v) >= 2]
    k = len(valid)
    if k < 3:
        raise ValueError("Welch ANOVA requires at least three groups.")

    n = np.array([len(v) for v in valid], dtype=float)
    means = np.array([np.mean(v) for v in valid], dtype=float)
    vars_ = np.array([np.var(v, ddof=1) for v in valid], dtype=float)
    vars_ = np.where(vars_ == 0, np.finfo(float).eps, vars_)

    w = n / vars_
    w_sum = np.sum(w)
    w_mean = np.sum(w * means) / w_sum

    num = np.sum(w * (means - w_mean) ** 2) / (k - 1)
    corr_term = np.sum((1 / (n - 1)) * (1 - w / w_sum) ** 2)
    correction = 1 + (2 * (k - 2) / (k**2 - 1)) * corr_term

    f_stat = num / correction
    df1, df2 = k - 1, (k**2 - 1) / (3 * corr_term)
    return f_stat, f.sf(f_stat, df1, df2), df1, df2


def generate_welch_anova(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = get_numeric_columns(), get_categorical_columns()
    logger.info("Computing Welch ANOVA tests...")
    results = []

    for nf in num_cols:
        for cf in cat_cols:
            try:
                groups = {
                    k: v for k, v in get_groups(df, nf, cf).items() if len(v) >= 2
                }
                if len(groups) < 3:
                    continue
                stat, p_val, df1, df2 = _welch_anova(groups)
                results.append(
                    {
                        "numeric_feature": nf,
                        "categorical_feature": cf,
                        "number_of_groups": len(groups),
                        "groups": " | ".join(groups.keys()),
                        "welch_f_statistic": stat,
                        "df1": df1,
                        "df2": df2,
                        "p_value": p_val,
                    }
                )
                logger.info(
                    f"Appended Welch ANOVA for ({nf}, {cf}) | F: {stat:.4f}, p: {p_val:.4e}"
                )
            except Exception as e:
                logger.warning(
                    f"Error computing Welch ANOVA for ({nf}, {cf}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# 5. Kruskal-Wallis Test
# ============================================================


def generate_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = get_numeric_columns(), get_categorical_columns()
    logger.info("Computing Kruskal-Wallis tests...")
    results = []

    for nf in num_cols:
        for cf in cat_cols:
            try:
                groups = {
                    k: v for k, v in get_groups(df, nf, cf).items() if len(v) > 0
                }
                if len(groups) < 3:
                    continue
                stat, p_val = kruskal(*groups.values())
                results.append(
                    {
                        "numeric_feature": nf,
                        "categorical_feature": cf,
                        "number_of_groups": len(groups),
                        "groups": " | ".join(groups.keys()),
                        "kruskal_h_statistic": stat,
                        "p_value": p_val,
                    }
                )
                logger.info(
                    f"Appended Kruskal-Wallis for ({nf}, {cf}) | H: {stat:.4f}, p: {p_val:.4e}"
                )
            except Exception as e:
                logger.warning(
                    f"Error computing Kruskal-Wallis for ({nf}, {cf}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# 6. Post-Hoc Analysis
# ============================================================


def generate_post_hoc(df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    num_cols, cat_cols = get_numeric_columns(), get_categorical_columns()
    logger.info("Computing post-hoc pairwise comparisons...")
    results = []

    for nf in num_cols:
        for cf in cat_cols:
            try:
                groups = {
                    k: v for k, v in get_groups(df, nf, cf).items() if len(v) > 0
                }
                if len(groups) < 3:
                    continue

                pairs = list(combinations(groups.keys(), 2))
                num_comps = len(pairs)

                for g1_name, g2_name in pairs:
                    g1, g2 = groups[g1_name], groups[g2_name]
                    stat, raw_p = mannwhitneyu(g1, g2, alternative="two-sided")
                    adj_p = min(raw_p * num_comps, 1.0)
                    results.append(
                        {
                            "numeric_feature": nf,
                            "categorical_feature": cf,
                            "group_1": g1_name,
                            "group_2": g2_name,
                            "n_group_1": len(g1),
                            "n_group_2": len(g2),
                            "u_statistic": stat,
                            "raw_p_value": raw_p,
                            "bonferroni_adjusted_p_value": adj_p,
                            "significant": adj_p < alpha,
                        }
                    )
                    logger.info(
                        f"Appended post-hoc comparison for ({nf}, {cf}) [{g1_name} vs {g2_name}] | adj_p: {adj_p:.4e}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error computing post-hoc analysis for ({nf}, {cf}): {e}"
                )

    return pd.DataFrame(results)


# ============================================================
# Effect Size Helpers & Function
# ============================================================


def _calculate_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    v1, v2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    return (
        (np.mean(g1) - np.mean(g2)) / np.sqrt(pooled_var)
        if pooled_var > 0
        else np.nan
    )


def _calculate_eta_squared(groups: dict[str, np.ndarray]) -> float:
    all_vals = np.concatenate(list(groups.values()))
    grand_mean = np.mean(all_vals)
    between_ss = sum(
        len(v) * (np.mean(v) - grand_mean) ** 2 for v in groups.values()
    )
    total_ss = np.sum((all_vals - grand_mean) ** 2)
    return between_ss / total_ss if total_ss > 0 else np.nan


def generate_effect_size(df: pd.DataFrame) -> pd.DataFrame:
    num_cols, cat_cols = get_numeric_columns(), get_categorical_columns()
    logger.info("Computing effect sizes...")
    results = []

    for nf in num_cols:
        for cf in cat_cols:
            try:
                groups = {
                    k: v for k, v in get_groups(df, nf, cf).items() if len(v) >= 2
                }
                if len(groups) == 2:
                    g_names = list(groups.keys())
                    eff_size = _calculate_cohens_d(
                        groups[g_names[0]], groups[g_names[1]]
                    )
                    results.append(
                        {
                            "numeric_feature": nf,
                            "categorical_feature": cf,
                            "effect_size_type": "cohens_d",
                            "group_1": g_names[0],
                            "group_2": g_names[1],
                            "number_of_groups": 2,
                            "effect_size": eff_size,
                        }
                    )
                    logger.info(
                        f"Appended Cohen's d for ({nf}, {cf}) | d: {eff_size:.4f}"
                    )
                elif len(groups) >= 3:
                    eff_size = _calculate_eta_squared(groups)
                    results.append(
                        {
                            "numeric_feature": nf,
                            "categorical_feature": cf,
                            "effect_size_type": "eta_squared",
                            "group_1": None,
                            "group_2": None,
                            "number_of_groups": len(groups),
                            "effect_size": eff_size,
                        }
                    )
                    logger.info(
                        f"Appended Eta-squared for ({nf}, {cf}) | eta²: {eff_size:.4f}"
                    )
            except Exception as e:
                logger.warning(
                    f"Error computing effect size for ({nf}, {cf}): {e}"
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


def run_num_cat_eda(df: pd.DataFrame) -> None:
    logger.info("Starting NUM-CAT exploratory data analysis...")
    results = {
        "group_statistics": generate_group_statistics(df),
        "welch_t_test": generate_welch_t_test(df),
        "mann_whitney_u": generate_mann_whitney_u(df),
        "welch_anova": generate_welch_anova(df),
        "kruskal_wallis": generate_kruskal_wallis(df),
        #"post_hoc": generate_post_hoc(df),
        "effect_size": generate_effect_size(df),
    }
    save_workbook(results)
    logger.info("NUM-CAT EDA pipeline completed successfully.")


def num_cat_eda(file_path: Path) -> None:
    logger.info(f"Attempting to load dataset from: {file_path}")
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        df = pd.read_csv(file_path)
        logger.info(f"Dataset successfully loaded with shape: {df.shape}")
        run_num_cat_eda(df)
    except (FileNotFoundError, pd.errors.EmptyDataError) as err:
        logger.error(err)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")
        raise


if __name__ == "__main__":
    num_cat_eda(file_path=BASE_DIR/ "src"/ "Data"/ "cleaned_ecommerce_dataset.csv")