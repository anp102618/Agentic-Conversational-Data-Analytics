from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from groq import Groq
from google import genai
from src.config import config


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class Config:
    groq_model: str = config.GROQ_MODEL
    gemini_model: str = config.GEMINI_MODEL
    output_report_file: str = r"D:\Conversational_Data_Analytics\src\Data_Prep\Quality_profiler\reports\data_quality_detailed_report.txt"
    output_json_file: str = r"D:\Conversational_Data_Analytics\src\Data_Prep\Quality_profiler\reports\data_quality_analysis.json"
    temperature: float = 0.1
    max_tokens: int = 1800
    request_delay_seconds: float = 10


CONFIG = Config()


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ColumnProfile:
    column_name: str
    physical_dtype: Optional[str]
    semantic_type: Optional[str]
    confidence: Optional[float]
    row_count: int
    non_null_count: int
    missing_count: int
    missing_percentage: float
    missing_severity: str
    unique_count: int


@dataclass
class ConsolidatedColumn:
    profile: ColumnProfile
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    cleaning_actions: List[Dict[str, Any]] = field(default_factory=list)
    dataset_level_anomalies: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class GeneratedSummary:
    column: str
    provider: Optional[str]
    model: Optional[str]
    status: str
    summary: str


# ============================================================================
# API CLIENTS
# ============================================================================

class LLMClients:
    """Initializes Groq and Gemini clients."""

    def __init__(self) -> None:
        groq_api_key = config.GROQ_API_KEY 
        gemini_api_key = config.GEMINI_API_KEY

        if not groq_api_key:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

        self.groq = Groq(api_key=groq_api_key)
        self.gemini = genai.Client(api_key=gemini_api_key)


# ============================================================================
# DATA QUALITY ANALYZER
# ============================================================================

class DataQualityAnalyzer:

    def __init__(self, config: Config, clients: LLMClients) -> None:
        self.config = config
        self.clients = clients

    @staticmethod
    def determine_missing_severity(missing_percentage: float) -> str:
        if missing_percentage == 0:
            return "none"
        if missing_percentage < 1:
            return "low"
        if missing_percentage < 10:
            return "moderate"
        return "high"

    def consolidate_columns(
        self, dataset_json: Dict[str, Any]
    ) -> Dict[str, ConsolidatedColumn]:

        columns = dataset_json.get("columns", [])
        anomalies = dataset_json.get("anomalies", [])
        cleaning_steps = dataset_json.get("cleaning_plan", {}).get("steps", [])
        relationships = dataset_json.get("relationships", [])

        dataset_level_anomalies = [
            a for a in anomalies if a.get("scope") == "dataset"
        ]

        consolidated: Dict[str, ConsolidatedColumn] = {}

        for raw_column in columns:
            column_name = raw_column["column"]
            missing_percentage = float(
                raw_column.get("missing_percentage", 0)
            )

            profile = ColumnProfile(
                column_name=column_name,
                physical_dtype=raw_column.get("physical_dtype"),
                semantic_type=raw_column.get("semantic_type"),
                confidence=raw_column.get("confidence"),
                row_count=int(raw_column.get("row_count", 0)),
                non_null_count=int(raw_column.get("non_null_count", 0)),
                missing_count=int(raw_column.get("missing_count", 0)),
                missing_percentage=missing_percentage,
                missing_severity=self.determine_missing_severity(
                    missing_percentage
                ),
                unique_count=int(raw_column.get("unique_count", 0)),
            )

            consolidated[column_name] = ConsolidatedColumn(
                profile=profile,
                anomalies=[
                    a for a in anomalies if a.get("scope") == column_name
                ],
                cleaning_actions=[
                    s for s in cleaning_steps if s.get("column") == column_name
                ],
                dataset_level_anomalies=dataset_level_anomalies,
                relationships=relationships,
            )

        return consolidated

    @staticmethod
    def consolidated_to_dict(
        consolidated: Dict[str, ConsolidatedColumn]
    ) -> Dict[str, Any]:
        return {
            name: asdict(info)
            for name, info in consolidated.items()
        }

    def build_column_prompt(
        self,
        column_name: str,
        column_info: ConsolidatedColumn,
        dataset_json: Dict[str, Any],
    ) -> str:

        return f"""
You are a senior data-quality analyst working on a machine-learning data preparation pipeline.

Analyze exactly ONE dataset column.

Your analysis must be based ONLY on the deterministic information provided below.

Do not invent:
- statistics
- distributions
- correlations
- values
- percentages
- categories
- business facts
- anomaly counts

If something is not present in the input, explicitly say that the information is not available.

============================================================
COLUMN
============================================================
{column_name}

============================================================
DETERMINISTIC COLUMN INFORMATION
============================================================
{json.dumps(asdict(column_info), indent=2, ensure_ascii=False)}

============================================================
DATASET INFORMATION
============================================================
{json.dumps(dataset_json.get("dataset", {}), indent=2, ensure_ascii=False)}

============================================================
OVERALL DATASET READINESS
============================================================
{json.dumps(dataset_json.get("readiness", {}), indent=2, ensure_ascii=False)}

============================================================
REQUIRED REPORT
============================================================

Generate a detailed report using exactly these sections:

1. Column Overview
2. Data Type and Semantic Interpretation
3. Completeness and Missingness
4. Cardinality and Uniqueness
5. Detected Data-Quality Issues
6. Recommended Cleaning Actions
7. Modeling Considerations
8. Risk Assessment
9. Final Column Verdict

============================================================
ANALYSIS RULES
============================================================

1. Explicitly mention:
   - row count
   - non-null count
   - missing count
   - missing percentage
   - unique count
   - physical dtype
   - semantic type

2. Discuss every supplied anomaly.
3. Discuss every supplied cleaning action.
4. Clearly distinguish observed facts, recommended actions, and modeling considerations.
5. Do not claim that a column has an anomaly unless explicitly listed.
6. If no column-specific anomalies exist, explicitly state:
   "No column-specific anomaly was detected in the supplied data-quality metadata."
7. If missing_count is zero, explicitly state that the column is complete.
8. If missing_count is greater than zero, discuss the supplied missingness objectively.
9. Do not calculate statistics that are not supplied.
10. Do not infer actual distributions.
11. Do not invent relationships.
12. Treat the cleaning plan as recommendations, not executed actions.
13. Keep the report useful for data cleaning, feature engineering, machine learning,
    data validation, and downstream analytics.
14. Use clear Markdown.
15. Be detailed but avoid unnecessary repetition.
"""

    def generate_with_groq(self, prompt: str) -> str:
        response = self.clients.groq.chat.completions.create(
            model=self.config.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a meticulous senior data-quality analyst. "
                        "You must use only supplied facts and must never "
                        "fabricate statistics."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")

        return content.strip()

    def generate_with_gemini(self, prompt: str) -> str:
        response = self.clients.gemini.models.generate_content(
            model=self.config.gemini_model,
            contents=prompt,
            config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            },
        )

        text = getattr(response, "text", None)
        if not text:
            raise RuntimeError("Gemini returned an empty response.")

        return text.strip()

    def generate_column_summary(
        self,
        column_name: str,
        column_info: ConsolidatedColumn,
        dataset_json: Dict[str, Any],
    ) -> GeneratedSummary:

        prompt = self.build_column_prompt(
            column_name, column_info, dataset_json
        )

        try:
            return GeneratedSummary(
                column=column_name,
                provider="groq",
                model=self.config.groq_model,
                status="success",
                summary=self.generate_with_groq(prompt),
            )
        except Exception as groq_error:
            print(f"[WARN] Groq failed for '{column_name}': {groq_error}")

        try:
            return GeneratedSummary(
                column=column_name,
                provider="gemini",
                model=self.config.gemini_model,
                status="success_fallback",
                summary=self.generate_with_gemini(prompt),
            )
        except Exception as gemini_error:
            print(
                f"[ERROR] Gemini fallback failed for "
                f"'{column_name}': {gemini_error}"
            )

            return GeneratedSummary(
                column=column_name,
                provider=None,
                model=None,
                status="failed",
                summary=(
                    "Column summary generation failed.\n\n"
                    f"Groq error: {groq_error}\n"
                    f"Gemini error: {gemini_error}"
                ),
            )

    def analyze_columns(
        self,
        dataset_json: Dict[str, Any],
        consolidated: Dict[str, ConsolidatedColumn],
    ) -> List[GeneratedSummary]:

        summaries: List[GeneratedSummary] = []
        total_columns = len(consolidated)

        for index, (column_name, column_info) in enumerate(
            consolidated.items(), start=1
        ):
            print(f"\n[{index}/{total_columns}] Analyzing: {column_name}")

            result = self.generate_column_summary(
                column_name, column_info, dataset_json
            )

            summaries.append(result)

            print(f"  Provider : {result.provider or 'none'}")
            print(f"  Status   : {result.status}")

            if (
                index < total_columns
                and self.config.request_delay_seconds > 0
            ):
                time.sleep(self.config.request_delay_seconds)

        return summaries


# ============================================================================
# REPORT BUILDER
# ============================================================================

class ReportBuilder:

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_text_report(
        self,
        dataset_json: Dict[str, Any],
        consolidated: Dict[str, ConsolidatedColumn],
        summaries: List[GeneratedSummary],
    ) -> str:

        dataset = dataset_json.get("dataset", {})
        readiness = dataset_json.get("readiness", {})

        lines: List[str] = []
        separator = "=" * 100
        sub_separator = "-" * 100

        lines += [
            separator,
            "DATA QUALITY DETAILED REPORT",
            separator,
            "",
            "1. DATASET OVERVIEW",
            sub_separator,
            f"Rows             : {dataset.get('rows', 'N/A')}",
            f"Columns          : {dataset.get('columns', 'N/A')}",
            f"Cells            : {dataset.get('cells', 'N/A')}",
            f"Readiness Score  : {readiness.get('score', 'N/A')}",
            f"Readiness Status : {readiness.get('status', 'N/A')}",
            f"Readiness Summary: {readiness.get('summary', 'N/A')}",
            "",
            "2. DETERMINISTIC COLUMN CONSOLIDATION",
            sub_separator,
        ]

        for column_name, column_info in consolidated.items():
            p = column_info.profile

            lines += [
                "",
                f"Column: {column_name}",
                f"  Physical dtype    : {p.physical_dtype}",
                f"  Semantic type     : {p.semantic_type}",
                f"  Confidence        : {p.confidence}",
                f"  Rows              : {p.row_count}",
                f"  Non-null          : {p.non_null_count}",
                f"  Missing           : {p.missing_count}",
                f"  Missing %         : {p.missing_percentage}",
                f"  Missing severity  : {p.missing_severity}",
                f"  Unique values     : {p.unique_count}",
                f"  Anomalies         : {len(column_info.anomalies)}",
                f"  Cleaning actions  : {len(column_info.cleaning_actions)}",
            ]

        lines += [
            "",
            "3. DETAILED COLUMN ANALYSIS",
            sub_separator,
        ]

        for result in summaries:
            lines += [
                "",
                separator,
                f"COLUMN: {result.column}",
                f"PROVIDER: {result.provider or 'N/A'}",
                f"MODEL: {result.model or 'N/A'}",
                f"STATUS: {result.status}",
                separator,
                result.summary,
            ]

        lines += [
            "",
            separator,
            "END OF DATA QUALITY REPORT",
            separator,
        ]

        return "\n".join(lines)

    def build_structured_output(
        self,
        dataset_json: Dict[str, Any],
        consolidated: Dict[str, ConsolidatedColumn],
        summaries: List[GeneratedSummary],
    ) -> Dict[str, Any]:

        return {
            "dataset": dataset_json.get("dataset"),
            "readiness": dataset_json.get("readiness"),
            "deterministic_consolidation": {
                name: asdict(info)
                for name, info in consolidated.items()
            },
            "generated_summaries": [
                asdict(summary) for summary in summaries
            ],
        }


# ============================================================================
# FILE PERSISTENCE
# ============================================================================

class ReportWriter:

    @staticmethod
    def write_text(path: str, content: str) -> None:
        Path(path).write_text(content, encoding="utf-8")

    @staticmethod
    def write_json(path: str, data: Dict[str, Any]) -> None:
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


# ============================================================================
# PIPELINE
# ============================================================================

class DataQualityPipeline:

    def __init__(self, config: Config) -> None:
        self.config = config
        self.clients = LLMClients()
        self.analyzer = DataQualityAnalyzer(config, self.clients)
        self.report_builder = ReportBuilder(config)
        self.report_writer = ReportWriter()

    def run(self, dataset_json: Dict[str, Any]) -> Dict[str, Any]:

        print("\n" + "=" * 100)
        print("STEP 1 - DETERMINISTIC COLUMN CONSOLIDATION")
        print("=" * 100)

        consolidated = self.analyzer.consolidate_columns(dataset_json)
        print(f"Consolidated {len(consolidated)} columns.")

        print("\n" + "=" * 100)
        print("STEP 2 - COLUMN-BY-COLUMN LLM ANALYSIS")
        print("=" * 100)

        summaries = self.analyzer.analyze_columns(
            dataset_json, consolidated
        )

        print("\n" + "=" * 100)
        print("STEP 3 - BUILDING FINAL REPORT")
        print("=" * 100)

        final_report = self.report_builder.build_text_report(
            dataset_json, consolidated, summaries
        )

        structured_output = self.report_builder.build_structured_output(
            dataset_json, consolidated, summaries
        )

        self.report_writer.write_text(
            self.config.output_report_file, final_report
        )

        self.report_writer.write_json(
            self.config.output_json_file, structured_output
        )

        print("\n" + "=" * 100)
        print("STEP 4 - FINAL REPORT")
        print("=" * 100)
        print("\n" + final_report)

        print("\n" + "=" * 100)
        print(f"Text report saved to: {self.config.output_report_file}")
        print(f"JSON report saved to: {self.config.output_json_file}")

        return {
            "consolidated": consolidated,
            "summaries": summaries,
            "final_report": final_report,
            "structured_output": structured_output,
        }


# ============================================================================
# INPUT LOADING
# ============================================================================

def load_json_file(file_path: str) -> Dict[str, Any]:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input JSON file does not exist: {file_path}"
        )

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("The root of the input JSON must be an object.")

    return data


# ============================================================================
# VALIDATION
# ============================================================================

def validate_dataset_json(dataset_json: Dict[str, Any]) -> None:

    required_sections = [
        "dataset",
        "columns",
        "anomalies",
        "relationships",
        "cleaning_plan",
        "readiness",
    ]

    missing_sections = [
        section for section in required_sections
        if section not in dataset_json
    ]

    if missing_sections:
        raise ValueError(
            "Input JSON is missing required sections: "
            + ", ".join(missing_sections)
        )

    if not isinstance(dataset_json["columns"], list):
        raise ValueError("'columns' must be a list.")

    if not isinstance(dataset_json["anomalies"], list):
        raise ValueError("'anomalies' must be a list.")


# ============================================================================
# MAIN
# ============================================================================

def summarize(filepath: str = r"D:\Conversational_Data_Analytics\src\Data_Prep\Quality_profiler\reports\pre_eda_report.json") -> None:

    input_file = filepath

    print("\n" + "=" * 100)
    print("DATA QUALITY ANALYSIS PIPELINE")
    print("=" * 100)

    print(f"\nLoading input: {input_file}")
    dataset_json = load_json_file(input_file)

    print("Validating input JSON...")
    validate_dataset_json(dataset_json)
    print("Input JSON validation successful.")

    pipeline = DataQualityPipeline(CONFIG)
    pipeline.run(dataset_json)



