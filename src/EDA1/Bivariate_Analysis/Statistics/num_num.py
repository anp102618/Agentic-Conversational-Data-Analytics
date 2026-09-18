from pathlib import Path
import logging
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.feature_selection import mutual_info_regression
from src.config import config, BASE_DIR

# ============================================================
# Configuration & Utilities
# ============================================================

OUTPUT_DIR = Path(BASE_DIR / "src" / "EDA1" / "Bivariate_Analysis" / "Reports")
WORKBOOK_NAME = "num_num_eda.xlsx"
LOG_FILE_NAME = "num_num_eda.log"

# ============================================================
# Simple Logging Configuration (Console + Appending File)
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_DIR / LOG_FILE_NAME

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns from the dataframe."""
    return df.select_dtypes(include="number").columns.tolist()

def get_valid_pair(df: pd.DataFrame, feature_1: str, feature_2: str) -> pd.DataFrame:
    """Return pairwise non-null observations for two features."""
    return df[[feature_1, feature_2]].dropna()

# ============================================================
# Correlation & Mutual Information Generators
# ============================================================

def _generate_correlation(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """Helper to generate Pearson, Spearman, or Kendall correlation results."""
    numeric_columns = get_numeric_columns(df)
    results = []
    func = {"pearson": pearsonr, "spearman": spearmanr, "kendall": kendalltau}[method]
    corr_key = "kendall_tau" if method == "kendall" else f"{method}_correlation"

    logger.info(f"Computing {method.capitalize()} correlations for {len(numeric_columns)} numeric features...")

    for i, f1 in enumerate(numeric_columns):
        for f2 in numeric_columns[i + 1:]:
            try:
                pair = get_valid_pair(df, f1, f2)
                if len(pair) < 3 or pair[f1].nunique() < 2 or pair[f2].nunique() < 2:
                    continue
                
                corr, p_val = func(pair[f1], pair[f2])
                result_entry = {
                    "feature_1": f1,
                    "feature_2": f2,
                    "sample_size": len(pair),
                    corr_key: corr,
                    "p_value": p_val,
                }
                results.append(result_entry)
                logger.info(f"Appended {method} correlation result for pair ({f1}, {f2}) | {corr_key}: {corr:.4f}, p_value: {p_val:.4e}")
            except Exception as e:
                logger.warning(f"Error computing {method} for pair ({f1}, {f2}): {e}")
                
    return pd.DataFrame(results)

def generate_mutual_information(df: pd.DataFrame) -> pd.DataFrame:
    """Generate mutual information between numeric feature pairs."""
    numeric_columns = get_numeric_columns(df)
    results = []

    logger.info(f"Computing Mutual Information for {len(numeric_columns)} numeric features...")

    for i, f1 in enumerate(numeric_columns):
        for f2 in numeric_columns[i + 1:]:
            try:
                pair = get_valid_pair(df, f1, f2)
                if len(pair) < 10 or pair[f2].nunique() < 2:
                    continue

                x, y = pair[f1].to_numpy().reshape(-1, 1), pair[f2].to_numpy()
                mi_xy = mutual_info_regression(x, y, random_state=42)[0]
                mi_yx = mutual_info_regression(y.reshape(-1, 1), pair[f1].to_numpy(), random_state=42)[0]
                mi_score = (mi_xy + mi_yx) / 2

                result_entry = {
                    "feature_1": f1,
                    "feature_2": f2,
                    "sample_size": len(pair),
                    "mutual_information": mi_score,
                }
                results.append(result_entry)
                logger.info(f"Appended mutual information result for pair ({f1}, {f2}) | MI: {mi_score:.4f}")
            except Exception as e:
                logger.warning(f"Error computing Mutual Information for pair ({f1}, {f2}): {e}")
                
    return pd.DataFrame(results)

# ============================================================
# Workbook Writer & Runner
# ============================================================

def save_workbook(results: dict[str, pd.DataFrame]) -> None:
    """Save all analysis results into one named Excel workbook."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        workbook_path = OUTPUT_DIR / WORKBOOK_NAME

        with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
            for sheet_name, dataframe in results.items():
                dataframe.to_excel(writer, sheet_name=sheet_name, index=False)
                
        logger.info(f"Workbook successfully created at: {workbook_path}")
    except Exception as e:
        logger.error(f"Failed to save Excel workbook: {e}")
        raise

def run_num_num_eda(df: pd.DataFrame) -> None:
    """Run all NUM-NUM analyses and create one workbook."""
    logger.info("Starting NUM-NUM exploratory data analysis...")
    results = {
        "pearson": _generate_correlation(df, "pearson"),
        "spearman": _generate_correlation(df, "spearman"),
        "kendall": _generate_correlation(df, "kendall"),
        "mutual_information": generate_mutual_information(df),
    }
    save_workbook(results)
    logger.info("NUM-NUM EDA pipeline completed successfully.")

def num_num_eda(file_path: Path) -> None:
    """Entry point execution flow with file loading safeguards."""
    logger.info(f"Attempting to load dataset from: {file_path}")
    try:
        if not file_path.exists():
            raise FileNotFoundError(f"The file {file_path} does not exist.")
            
        dataframe = pd.read_csv(file_path)
        logger.info(f"Dataset successfully loaded with shape: {dataframe.shape}")
        run_num_num_eda(dataframe)
        
    except FileNotFoundError as fnf_error:
        logger.error(fnf_error)
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"The dataset at {file_path} is empty.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during execution: {e}")
        raise

if __name__ == "__main__":
    num_num_eda(file_path=Path(BASE_DIR / "src" / "Data" / "cleaned_ecommerce_dataset.csv"))