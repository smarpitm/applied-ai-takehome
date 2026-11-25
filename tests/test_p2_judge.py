"""Unit & integration tests for Problem 2 LLM-as-Judge schemas, parser, biases, validation, A/B."""

from __future__ import annotations

import json
from pathlib import Path

from problem2_judge.comparator import (
    binomial_sign_test_p_value,
    bootstrap_mean_diff_ci,
    run_ab_comparison,
)
from problem2_judge.judge_engine import JudgeEngine
from problem2_judge.parser import parse_verdict_with_recovery
from problem2_judge.schemas import CriterionVerdict, PointwiseVerdict, TestCase
from problem2_judge.validation import compute_cohens_kappa
from shared.config import Settings
from shared.llm_client import LLMRouter
from shared.logging_utils import read_jsonl


def _sample_pointwise_dict(score: int = 5, safety: int = 5) -> dict:
    block = {
        "quoted_evidence": ["exact quote from candidate"],
        "rationale": "Matches expected output and adheres to all rules.",
        "score": score,
    }
    return {
        "criteria": {
            "correctness": block,
            "faithfulness": block,
            "completeness": block,
            "instruction_following": block,
            "conciseness": block,
            "safety": {**block, "score": safety},
        },
        "overall_rationale": "Strong grounded output.",
    }


def _sample_pairwise_dict(winner: str = "A", sa: int = 5, sb: int = 3) -> dict:
    crit = {
        "quoted_evidence_a": ["quote A"],
        "quoted_evidence_b": ["quote B"],
        "rationale": "Candidate A is factually accurate and concise.",
        "score_a": sa,
        "score_b": sb,
    }
    return {
        "criteria": {
            "correctness": crit,
            "faithfulness": crit,
            "completeness": crit,
            "instruction_following": crit,
            "conciseness": crit,
            "safety": crit,
        },
        "overall_rationale": "Winner selected from criterion evidence.",
        "winner": winner,
    }


def test_evidence_before_score_field_order() -> None:
    fields = list(CriterionVerdict.model_fields.keys())
    assert fields.index("quoted_evidence") < fields.index("rationale") < fields.index("score")


def test_three_stage_parser_recovery() -> None:
    payload = _sample_pointwise_dict()
    valid_json = json.dumps(payload)

    # Stage 1: Direct JSON
    r1 = parse_verdict_with_recovery(valid_json, PointwiseVerdict)
    assert r1.parse_stage == "direct" and not r1.parse_error

    # Stage 2: Surrounding prose + markdown fence + trailing comma
    messy = f"<think>evaluating</think>\nHere is the verdict:\n```json\n{valid_json[:-1]},}}\n```"
    r2 = parse_verdict_with_recovery(messy, PointwiseVerdict)
    assert r2.parse_stage == "regex_extract" and not r2.parse_error

    # Stage 3: Invalid text repaired via 1 retry
    r3 = parse_verdict_with_recovery(
        "broken json {",
        PointwiseVerdict,
        repair_fn=lambda prompt: valid_json,
    )
    assert r3.parse_stage == "repair_retry" and r3.retried and not r3.parse_error

    # Failed gracefully without crashing
    r4 = parse_verdict_with_recovery(
        "still broken",
        PointwiseVerdict,
        repair_fn=lambda prompt: "nope",
    )
    assert r4.parse_stage == "failed" and r4.parse_error is True and r4.verdict is None


def test_bidirectional_position_flip_and_audit_trace(tmp_path: Path) -> None:
    # Judge always picks whichever candidate is presented FIRST ("A") -> 100% flip rate!
    def position_biased_transport(
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        return json.dumps(_sample_pairwise_dict(winner="A", sa=5, sb=3)), 100, 50

    settings = Settings(gemini_api_key="dummy", log_dir=tmp_path / "logs")
    router = LLMRouter(settings=settings, transport=position_biased_transport)
    audit_path = tmp_path / "logs" / "trace.jsonl"
    engine = JudgeEngine(router=router, settings=settings, audit_log_path=audit_path)

    case = TestCase(
        id="pair_1",
        input="What is 2+2?",
        output_a="4",
        output_b="4.",
    )
    rep, results = engine.run_pairwise_suite([case], suite_name="flip_test", swap_orders=True)
    assert rep.position_flip_rate == 1.0
    assert results[0].position_flipped is True
    assert results[0].reconciled_winner == "tie"
    assert rep.ties == 1

    traces = read_jsonl(audit_path)
    assert len(traces) == 2
    assert {t["presentation_order"] for t in traces} == {"AB", "BA"}


def test_cohen_kappa_sign_test_and_ab_gate(tmp_path: Path) -> None:
    assert compute_cohens_kappa([True, True, False, False], [True, True, False, False]) == 1.0
    assert binomial_sign_test_p_value(0, 12) < 0.001
    ci_lo, ci_hi = bootstrap_mean_diff_ci([1.0, 1.2, 0.8, 1.1, 0.9, 1.3])
    assert ci_lo > 0.0 and ci_hi > ci_lo

    # Simulate Config B winning cleanly in both AB (picks B) and BA (picks A -> mapped B)
    def pro_b_transport(
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        user_content = messages[-1]["content"]
        if "Mode: Pointwise" in user_content:
            is_b = "48 bytes [c1]" in user_content
            return json.dumps(_sample_pointwise_dict(score=5 if is_b else 3)), 90, 40
        # Pairwise: pick whichever candidate slot has "[c1]"
        part_a = user_content.split("--- Candidate A ---")[1].split("--- Candidate B ---")[0]
        winner = "A" if "[c1]" in part_a else "B"
        sa, sb = (5, 3) if winner == "A" else (3, 5)
        return json.dumps(_sample_pairwise_dict(winner=winner, sa=sa, sb=sb)), 110, 55

    settings = Settings(gemini_api_key="dummy", log_dir=tmp_path / "logs")
    router = LLMRouter(settings=settings, transport=pro_b_transport)
    engine = JudgeEngine(router=router, settings=settings, audit_log_path=tmp_path / "audit.jsonl")

    ab_data = {
        "config_a": {"name": "prompt_v1", "system_prompt": "Answer."},
        "config_b": {"name": "prompt_v2", "system_prompt": "Cite."},
        "cases": [
            {
                "id": f"ab_{i}",
                "input": "Size?",
                "expected_output": "48 bytes [c1]",
                "output_a": "Small.",
                "output_b": "48 bytes [c1]",
            }
            for i in range(8)
        ],
    }
    ab_rep = run_ab_comparison(ab_data, engine=engine, settings=settings)
    assert ab_rep["declared_winner"] == "prompt_v2"
    assert ab_rep["position_flip_rate"] == 0.0
    assert ab_rep["binomial_sign_test_p_value"] < 0.01
