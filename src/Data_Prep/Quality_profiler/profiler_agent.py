from __future__ import annotations

import json
import re
import time

import pandas as pd
from google import genai
from groq import Groq
from pydantic import ValidationError

from .quality_tools import DeterministicProfiler
from .schemas import (
    Anomaly,
    CleaningPlan,
    CleaningStep,
    ColumnInsight,
    ColumnProfile,
    DatasetInfo,
    PreEDAReport,
    Readiness,
    Relationship,
)
from src.config import config


class PreEDAAgent:
    def __init__(
        self,
        df: pd.DataFrame,
        groq_model: str = config.GROQ_MODEL,
        gemini_model: str = config.GEMINI_MODEL,
        delay_seconds: int = 10,
        target_column: str | None = None,
        categorical_allowed_values: dict[str, set[str]] | None = None,
    ):
        self.df = df
        self.groq_model = groq_model
        self.gemini_model = gemini_model
        self.delay_seconds = delay_seconds

        if not config.GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY environment variable is not set.")
        if not config.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

        self.groq_client = Groq(api_key=config.GROQ_API_KEY)
        self.gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
        self.profiler = DeterministicProfiler(
            df=df,
            target_column=target_column,
            categorical_allowed_values=categorical_allowed_values,
        )

    def run(self) -> dict:
        print("[INFO] Running deterministic profiling...")
        evidence = self.profiler.build_evidence()
        dataset, profiles = evidence["dataset"], evidence["columns"]

        print(f"[INFO] Dataset: {dataset['rows']} rows x {dataset['columns']} columns")
        print(f"[INFO] Starting AI column analysis: {len(profiles)} columns")

        insights = []

        for i, profile in enumerate(profiles, 1):
            column = profile["column"]
            print(f"[INFO] Column {i}/{len(profiles)}: {column}")

            try:
                insight = self._analyze_column(profile)
                print(f"[INFO] AI success: {column}")
            except Exception as exc:
                print(f"[WARNING] AI analysis failed for {column}: {exc}")
                insight = self._fallback_column_insight(profile)

            insights.append(insight)

            if i < len(profiles) and self.delay_seconds > 0:
                print(f"[INFO] Waiting {self.delay_seconds}s...")
                time.sleep(self.delay_seconds)

        report = self._build_final_report(
            dataset,
            profiles,
            insights,
            evidence["duplicate_rows"],
        )
        validated = self._validate_final_report(report)
        print("[INFO] Final report validated successfully.")
        return validated

    # -----------------------------------------------------
    # AI
    # -----------------------------------------------------

    def _analyze_column(self, evidence: dict) -> ColumnInsight:
        try:
            print(f"[INFO] Trying Groq: {self.groq_model}")
            return self._analyze_with_groq(evidence)
        except Exception as exc:
            print(f"[WARNING] Groq failed: {exc}")

        try:
            print(f"[INFO] Trying Gemini backup: {self.gemini_model}")
            return self._analyze_with_gemini(evidence)
        except Exception as exc:
            print(f"[WARNING] Gemini failed: {exc}")

        print("[INFO] Using deterministic fallback.")
        return self._fallback_column_insight(evidence)

    def _analyze_with_groq(self, evidence: dict) -> ColumnInsight:
        response = self.groq_client.chat.completions.create(
            model=self.groq_model,
            messages=[
                {"role": "system", "content": self._column_system_prompt()},
                {"role": "user", "content": self._column_prompt(evidence)},
            ],
            temperature=0,
            max_tokens=1000,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned empty response.")

        return ColumnInsight.model_validate(self._extract_json(content))

    def _analyze_with_gemini(self, evidence: dict) -> ColumnInsight:
        prompt = self._column_system_prompt() + "\n\n" + self._column_prompt(evidence)

        response = self.gemini_client.models.generate_content(
            model=self.gemini_model,
            contents=prompt,
            config={
                "temperature": 0,
                "response_mime_type": "application/json",
            },
        )

        if not response.text:
            raise ValueError("Gemini returned empty response.")

        try:
            data = self._extract_json(response.text)
        except (ValueError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Gemini returned invalid JSON: {response.text[:500]}"
            ) from exc

        try:
            return ColumnInsight.model_validate(data)
        except ValidationError as exc:
            raise ValueError(
                f"Gemini returned invalid ColumnInsight:\n{exc}"
            ) from exc

    # -----------------------------------------------------
    # PROMPTS
    # -----------------------------------------------------

    def _column_system_prompt(self) -> str:
        return """
You are a professional data-quality analyst.

Analyze exactly ONE dataset column using only the supplied
deterministic evidence.

Return ONLY valid JSON:

{
  "column": "string",
  "semantic_type": "string",
  "confirmed_issues": [],
  "potential_anomalies": [],
  "cleaning_recommendations": [],
  "decomposition_needed": false,
  "decomposition_reason": null,
  "suggested_columns": []
}

Rules:
- Never invent statistics or unsupported issues.
- confirmed_issues may only contain:
  missing_values, duplicate_rows, outliers, invalid_values,
  constant_column, near_constant_column, high_cardinality,
  id_column, class_imbalance, potential_leakage,
  datetime_issues, invalid_categorical_values
- Maximum 3 confirmed issues.
- Maximum 3 potential anomalies.
- Maximum 3 cleaning recommendations.
- Do not add fields.
- No markdown.

Set decomposition_needed=true only when a string/categorical
value contains multiple independent concepts, e.g.
"Male|32|Delhi" -> gender, age, city
"Premium - Active - India" -> plan, status, country
"Electronics > Laptop > Gaming" -> department, category, subcategory

Do not decompose ordinary categorical values such as
"Male", "Female", "Other".

If uncertain, set decomposition_needed=false and use
"investigate" in cleaning_recommendations.
"""

    def _column_prompt(self, evidence: dict) -> str:
        return (
            "Deterministic column evidence:\n\n"
            + json.dumps(evidence, separators=(",", ":"), default=str)
        )

    # -----------------------------------------------------
    # JSON
    # -----------------------------------------------------

    def _extract_json(self, content: str) -> dict:
        if not content:
            raise ValueError("AI response is empty.")

        content = content.strip()
        content = re.sub(r"^```json\s*", "", content, flags=re.I)
        content = re.sub(r"^```\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

        try:
            data = json.loads(content)
            if not isinstance(data, dict):
                raise ValueError("AI JSON response must be an object.")
            return data
        except json.JSONDecodeError:
            pass

        start, end = content.find("{"), content.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("AI response does not contain a JSON object.")

        try:
            data = json.loads(content[start:end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("AI response contains malformed JSON.") from exc

        if not isinstance(data, dict):
            raise ValueError("AI JSON response must be an object.")

        return data

    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    def _fallback_column_insight(self, evidence: dict) -> ColumnInsight:
        issues = []
        anomalies = []
        recommendations = []
        deterministic = evidence.get("issues", {})
        issue_names = set(deterministic)

        for issue in (
            "missing_values",
            "outliers",
            "invalid_values",
            "constant_column",
            "near_constant_column",
            "high_cardinality",
            "id_column",
            "class_imbalance",
            "potential_leakage",
            "datetime_issues",
            "invalid_categorical_values",
        ):
            if issue in issue_names:
                issues.append(issue)

        recommendations_map = {
            "missing_values": "Review missing-value handling.",
            "outliers": "Investigate detected numeric outliers.",
            "invalid_values": "Validate and correct invalid values.",
            "invalid_categorical_values":
                "Validate categorical values against the allowed categories.",
            "constant_column": "Consider removing the constant column.",
            "near_constant_column":
                "Review whether the near-constant column provides useful information.",
            "high_cardinality":
                "Review high-cardinality encoding and modeling strategy.",
            "id_column":
                "Review whether this identifier should be excluded from modeling.",
            "potential_leakage":
                "Investigate potential target leakage.",
            "datetime_issues":
                "Validate datetime parsing and values.",
            "class_imbalance":
                "Review class imbalance before modeling.",
        }

        for issue in issue_names:
            if issue in recommendations_map:
                recommendations.append(recommendations_map[issue])

        if evidence.get("has_leading_spaces"):
            issues.append("invalid_values")
            recommendations.append("Trim leading whitespace.")

        if evidence.get("has_trailing_spaces"):
            issues.append("invalid_values")
            recommendations.append("Trim trailing whitespace.")

        if evidence.get("has_multiple_spaces"):
            anomalies.append("Multiple consecutive spaces detected.")
            recommendations.append("Normalize whitespace.")

        decomposition = self._detect_decomposition(evidence)
        suggested = (
            self._suggest_decomposition_columns(evidence)
            if decomposition
            else []
        )

        if decomposition:
            anomalies.append(
                "Categorical values may contain multiple embedded concepts."
            )
            recommendations.append(
                "Decompose the categorical field into separate columns."
            )

        return ColumnInsight(
            column=evidence["column"],
            semantic_type=evidence["semantic_type"],
            confirmed_issues=list(dict.fromkeys(issues))[:3],
            potential_anomalies=list(dict.fromkeys(anomalies))[:3],
            cleaning_recommendations=list(
                dict.fromkeys(recommendations)
            )[:3],
            decomposition_needed=decomposition,
            decomposition_reason=(
                "Values appear to contain multiple embedded attributes."
                if decomposition else None
            ),
            suggested_columns=suggested[:5],
        )

    # -----------------------------------------------------
    # DECOMPOSITION
    # -----------------------------------------------------

    def _detect_decomposition(self, evidence: dict) -> bool:
        if evidence.get("semantic_type") not in {"categorical", "string"}:
            return False

        values = list(evidence.get("top_values", {}).keys())
        if not values:
            return False

        delimiters = ("|", ";", " - ", " > ", " / ", ",", ":")
        return any(
            sum(d in value for value in values) >= 2
            for d in delimiters
        )

    def _suggest_decomposition_columns(self, evidence: dict) -> list[str]:
        values = list(evidence.get("top_values", {}).keys())
        if not values:
            return []

        delimiter = next(
            (
                d for d in ("|", ";", " - ", " > ", " / ", ",", ":")
                if sum(d in value for value in values) >= 2
            ),
            None,
        )

        if delimiter is None:
            return []

        max_parts = max(len(value.split(delimiter)) for value in values)
        return [
            f"{evidence['column']}_part_{i}"
            for i in range(1, max_parts + 1)
        ]

    # -----------------------------------------------------
    # FINAL REPORT
    # -----------------------------------------------------

    def _build_final_report(
        self,
        dataset: dict,
        profiles: list[dict],
        insights: list[ColumnInsight],
        duplicate_rows: int,
    ) -> dict:
        columns = [
            self._build_column_profile(p, i).model_dump()
            for p, i in zip(profiles, insights)
        ]

        anomalies = self._build_anomalies(
            profiles, insights, duplicate_rows
        )

        return {
            "dataset": DatasetInfo(**dataset).model_dump(),
            "columns": columns,
            "anomalies": [a.model_dump() for a in anomalies],
            "relationships": [
                r.model_dump()
                for r in self._build_relationships(profiles)
            ],
            "cleaning_plan": self._build_cleaning_plan(
                profiles, insights, duplicate_rows
            ).model_dump(),
            "readiness": self._calculate_readiness(
                anomalies
            ).model_dump(),
        }

    def _build_column_profile(
        self,
        profile: dict,
        insight: ColumnInsight,
    ) -> ColumnProfile:
        return ColumnProfile(
            column=profile["column"],
            physical_dtype=profile["physical_dtype"],
            row_count=profile["row_count"],
            non_null_count=profile["non_null_count"],
            missing_count=profile.get("missing_count", 0),
            missing_percentage=profile.get("missing_percentage", 0.0),
            unique_count=profile.get("unique_count", 0),
            semantic_type=insight.semantic_type,
            confidence=1.0,
        )

    # -----------------------------------------------------
    # ANOMALIES
    # -----------------------------------------------------

    def _build_anomalies(
        self,
        profiles: list[dict],
        insights: list[ColumnInsight],
        duplicate_rows: int,
    ) -> list[Anomaly]:
        anomalies = []

        if duplicate_rows > 0:
            anomalies.append(
                Anomaly(
                    scope="dataset",
                    anomaly_type="duplicate_rows",
                    description=f"{duplicate_rows} duplicate rows detected.",
                    severity="high",
                )
            )

        for profile, _ in zip(profiles, insights):
            for issue_type in profile.get("issues", {}):
                severity = (
                    "medium"
                    if issue_type in {
                        "missing_values",
                        "outliers",
                        "class_imbalance",
                        "datetime_issues",
                    }
                    else "high"
                    if issue_type in {
                        "invalid_values",
                        "invalid_categorical_values",
                        "potential_leakage",
                    }
                    else "low"
                )

                anomalies.append(
                    Anomaly(
                        scope=profile["column"],
                        anomaly_type=issue_type,
                        description=self._issue_description(issue_type),
                        severity=severity,
                    )
                )

        return anomalies

    def _issue_description(self, issue_type: str) -> str:
        return {
            "missing_values":
                "Missing values detected in this column.",
            "outliers":
                "Statistical outliers detected using the IQR method.",
            "invalid_values":
                "Values failed deterministic validation or normalization checks.",
            "constant_column":
                "The column contains only one non-null value.",
            "near_constant_column":
                "One value dominates the column, making it near-constant.",
            "high_cardinality":
                "The column has high cardinality relative to its row count.",
            "id_column":
                "The column appears to be an identifier.",
            "class_imbalance":
                "The target column has a strongly imbalanced class distribution.",
            "potential_leakage":
                "The column may contain information derived from or strongly related to the target.",
            "datetime_issues":
                "Potential datetime parsing or validity issues were detected.",
            "invalid_categorical_values":
                "Categorical values outside the configured allowed set were detected.",
        }.get(issue_type, "Data-quality issue detected.")

    # -----------------------------------------------------
    # CLEANING
    # -----------------------------------------------------

    def _build_cleaning_plan(
        self,
        profiles: list[dict],
        insights: list[ColumnInsight],
        duplicate_rows: int,
    ) -> CleaningPlan:
        steps = []

        if duplicate_rows > 0:
            steps.append(
                CleaningStep(
                    column=None,
                    action="remove_duplicates",
                    reason=f"{duplicate_rows} duplicate rows were detected.",
                )
            )

        for profile, insight in zip(profiles, insights):
            column = profile["column"]

            if insight.decomposition_needed:
                steps.append(
                    CleaningStep(
                        column=column,
                        action="decompose",
                        reason=(
                            insight.decomposition_reason
                            or "Column appears to contain multiple concepts."
                        )
                        + " Suggested columns: "
                        + ", ".join(insight.suggested_columns),
                    )
                )

            for recommendation in insight.cleaning_recommendations:
                steps.append(
                    CleaningStep(
                        column=column,
                        action=self._cleaning_action(recommendation),
                        reason=recommendation,
                    )
                )

        return CleaningPlan(steps=steps)

    def _cleaning_action(self, recommendation: str) -> str:
        text = recommendation.lower()

        rules = [
            (
                ("decompos", "split into", "separate columns"),
                "decompose",
            ),
            (("duplicate",), "remove_duplicates"),
            (("convert", "parse", "numeric type", "datetime"), "convert_type"),
            (("trim", "whitespace", "normalize", "standardize"), "standardize"),
            (("investigate", "investigation"), "investigate"),
            (("validate", "validation"), "validate"),
            (("review", "consider"), "review"),
        ]

        for terms, action in rules:
            if any(term in text for term in terms):
                return action

        return "defer"

    # -----------------------------------------------------
    # RELATIONSHIPS
    # -----------------------------------------------------

    def _build_relationships(
        self,
        profiles: list[dict],
    ) -> list[Relationship]:
        numeric = [
            p["column"]
            for p in profiles
            if p["semantic_type"] in {"numeric", "numeric_string"}
        ]

        if len(numeric) < 2:
            return []

        return [
            Relationship(
                relationship="numeric_columns",
                details=(
                    "Multiple numeric columns are available for "
                    "correlation and business-rule analysis."
                ),
            )
        ]

    # -----------------------------------------------------
    # READINESS
    # -----------------------------------------------------

    def _calculate_readiness(
        self,
        anomalies: list[Anomaly],
    ) -> Readiness:
        if not anomalies:
            return Readiness(
                score=100.0,
                status="ready",
                summary="No deterministic data-quality issues were detected.",
            )

        high = sum(a.severity == "high" for a in anomalies)
        medium = sum(a.severity == "medium" for a in anomalies)
        low = sum(a.severity == "low" for a in anomalies)

        score = max(
            0.0,
            min(100.0, 100.0 - high * 10 - medium * 5 - low * 2),
        )

        status = (
            "mostly_ready"
            if score >= 80
            else "needs_cleaning"
            if score >= 50
            else "not_ready"
        )

        return Readiness(
            score=round(score, 2),
            status=status,
            summary=(
                f"Detected {len(anomalies)} deterministic "
                "data-quality observations."
            ),
        )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    def _validate_final_report(self, report: dict) -> dict:
        try:
            return PreEDAReport.model_validate(report).model_dump(mode="json")
        except ValidationError as exc:
            raise ValueError(
                f"Final PreEDAReport validation failed:\n{exc}"
            ) from exc


    # -----------------------------------------------------
    #EXECUTE PIPELINE
    # -----------------------------------------------------

    def run_pre_eda(file_path: str, output_path: str = r"D:\Conversational_Data_Analytics\src\Data_Prep\Quality_profiler\reports\pre_eda_report.json"):
        """Run pre-EDA profiling on a CSV file and return the validated report."""

        print(f"[INFO] Loading: {file_path}")
        df = pd.read_csv(file_path)
        print(f"[INFO] Dataset: {len(df):,} rows x {len(df.columns):,} columns")

        report = PreEDAAgent(df).run()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        print(f"[INFO] Report saved: {output_path}")
        return report