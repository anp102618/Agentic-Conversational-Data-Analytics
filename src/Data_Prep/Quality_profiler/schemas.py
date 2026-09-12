from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# =========================================================
# ISSUE TYPES
# =========================================================

IssueType = Literal[
    "missing_values",
    "duplicate_rows",
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
]

SemanticType = Literal[
    "numeric",
    "numeric_string",
    "categorical",
    "string",
    "datetime",
    "datetime_string",
    "unknown",
]


# =========================================================
# COLUMN-LEVEL AI OUTPUT
# =========================================================

class ColumnInsight(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    semantic_type: str

    confirmed_issues: list[IssueType] = Field(
        default_factory=list
    )

    potential_anomalies: list[str] = Field(
        default_factory=list
    )

    cleaning_recommendations: list[str] = Field(
        default_factory=list
    )

    decomposition_needed: bool = False

    decomposition_reason: str | None = None

    suggested_columns: list[str] = Field(
        default_factory=list
    )


# =========================================================
# FINAL REPORT
# =========================================================

class DatasetInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rows: int
    columns: int
    cells: int


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str
    physical_dtype: str
    row_count: int
    non_null_count: int

    missing_count: int = 0
    missing_percentage: float = 0.0
    unique_count: int = 0

    semantic_type: str
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
    )


class Anomaly(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: str

    anomaly_type: IssueType

    description: str

    severity: Literal[
        "low",
        "medium",
        "high",
    ] = "medium"


class Relationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship: str
    details: str


class CleaningStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str | None = None

    action: Literal[
        "keep",
        "standardize",
        "convert_type",
        "investigate",
        "remove_duplicates",
        "decompose",
        "validate",
        "review",
        "defer",
    ]

    reason: str


class CleaningPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[CleaningStep] = Field(
        default_factory=list
    )


class Readiness(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )

    status: Literal[
        "ready",
        "mostly_ready",
        "needs_cleaning",
        "not_ready",
    ] = "needs_cleaning"

    summary: str = ""


class PreEDAReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset: DatasetInfo

    columns: list[ColumnProfile]

    anomalies: list[Anomaly] = Field(
        default_factory=list
    )

    relationships: list[Relationship] = Field(
        default_factory=list
    )

    cleaning_plan: CleaningPlan

    readiness: Readiness
