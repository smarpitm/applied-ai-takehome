"""Pydantic v2 schemas for Problem 2 LLM-as-Judge pipeline.

Note: In every criterion verdict model, `quoted_evidence` and `rationale` are
intentionally ordered BEFORE `score` so autoregressive left-to-right token
generation forces evidence extraction prior to assigning any numeric rating.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CRITERION_NAMES: tuple[str, ...] = (
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "conciseness",
    "safety",
)


class CriterionVerdict(BaseModel):
    """Evidence-first single-output criterion verdict."""

    quoted_evidence: list[str] = Field(
        default_factory=list,
        description="Verbatim quotes from candidate output or context supporting the evaluation.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="Grounded reasoning verifying accuracy/constraints before assigning score.",
    )
    score: int = Field(..., ge=1, le=5, description="Integer rubric score from 1 to 5.")


class PointwiseCriteriaBlock(BaseModel):
    """All six explicit evaluation criteria for pointwise mode."""

    correctness: CriterionVerdict
    faithfulness: CriterionVerdict
    completeness: CriterionVerdict
    instruction_following: CriterionVerdict
    conciseness: CriterionVerdict
    safety: CriterionVerdict


class PointwiseVerdict(BaseModel):
    """Structured pointwise judge verdict (rationale precedes score)."""

    criteria: PointwiseCriteriaBlock
    overall_rationale: str = Field(..., min_length=1)
    overall_score: float | None = Field(default=None, ge=1.0, le=5.0)
    passed: bool | None = None


class PairwiseCriterionVerdict(BaseModel):
    """Evidence-first side-by-side criterion evaluation (quotes + rationale before scores)."""

    quoted_evidence_a: list[str] = Field(default_factory=list)
    quoted_evidence_b: list[str] = Field(default_factory=list)
    rationale: str = Field(..., min_length=1)
    score_a: int = Field(..., ge=1, le=5)
    score_b: int = Field(..., ge=1, le=5)


class PairwiseCriteriaBlock(BaseModel):
    """All six explicit evaluation criteria for pairwise A-vs-B mode."""

    correctness: PairwiseCriterionVerdict
    faithfulness: PairwiseCriterionVerdict
    completeness: PairwiseCriterionVerdict
    instruction_following: PairwiseCriterionVerdict
    conciseness: PairwiseCriterionVerdict
    safety: PairwiseCriterionVerdict


class PairwiseVerdict(BaseModel):
    """Structured pairwise A-vs-B judge verdict (rationale precedes winner)."""

    criteria: PairwiseCriteriaBlock
    overall_rationale: str = Field(..., min_length=1)
    winner: Literal["A", "B", "tie"]


class TestCase(BaseModel):
    """Input test case schema supporting single-output and pairwise A-vs-B suites."""

    __test__ = False

    id: str
    category: str = "general"
    input: str
    system_prompt: str = ""
    expected_output: str | None = None
    context: str | None = None
    criteria: list[str] | None = None
    model_output: str | None = None
    output_a: str | None = None
    output_b: str | None = None
    gold_pass: bool | None = None
    gold_winner: Literal["A", "B", "tie"] | None = None


class AuditTraceRecord(BaseModel):
    """Append-only JSONL audit record written for every single judge LLM call."""

    timestamp: str
    run_id: str
    case_id: str
    mode: Literal["pointwise", "pairwise"]
    presentation_order: Literal["POINTWISE", "AB", "BA"]
    judge_variant: Literal["mitigated", "baseline"]
    provider_requested: str
    provider_used: str
    model_requested: str
    model_used: str
    fallback_triggered: bool
    full_prompt: str
    raw_response: str
    parsed_verdict: dict[str, Any] | None = None
    parse_stage: str
    retried: bool
    latency_ms: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float


class ReconciledPairResult(BaseModel):
    """Position-bias reconciled result for one pair across AB and BA presentations."""

    case_id: str
    raw_winner_ab: Literal["A", "B", "tie", "error"]
    raw_winner_ba_mapped: Literal["A", "B", "tie", "error"]
    position_flipped: bool
    reconciled_winner: Literal["A", "B", "tie"]
    mean_score_a: float
    mean_score_b: float
    criterion_means_a: dict[str, float]
    criterion_means_b: dict[str, float]
    chars_a: int
    chars_b: int
    tokens_a: int
    tokens_b: int
    safety_or_correctness_regression_b: bool


class SuiteReport(BaseModel):
    """Aggregate report emitted at the end of a suite evaluation."""

    run_id: str
    suite_name: str
    mode: Literal["pointwise", "pairwise"]
    total_cases: int
    pass_rate: float | None = None
    mean_overall_score: float | None = None
    criterion_means: dict[str, float]
    score_std_dev: float | None = None
    score_distribution: dict[str, int] = Field(default_factory=dict)
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    position_flip_rate: float | None = None
    parse_error_count: int = 0
    retry_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    extra: dict[str, Any] = Field(default_factory=dict)
