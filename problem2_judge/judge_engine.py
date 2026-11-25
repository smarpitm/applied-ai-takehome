"""Unified pointwise & pairwise A-vs-B LLM judge engine with append-only JSONL trace."""

from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import tiktoken

from problem2_judge.parser import ParseResult, parse_verdict_with_recovery
from problem2_judge.rubrics import (
    RUBRIC_WEIGHTS,
    build_pairwise_prompt,
    build_pointwise_prompt,
    compute_weighted_score,
)
from problem2_judge.schemas import (
    CRITERION_NAMES,
    AuditTraceRecord,
    PairwiseVerdict,
    PointwiseVerdict,
    ReconciledPairResult,
    SuiteReport,
    TestCase,
)
from shared.config import ModelEndpointConfig, Settings, get_settings
from shared.llm_client import LLMResponse, LLMRouter
from shared.logging_utils import append_jsonl, utc_now_iso

_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text or ""))


def _flip_winner(winner: Literal["A", "B", "tie"]) -> Literal["A", "B", "tie"]:
    if winner == "A":
        return "B"
    if winner == "B":
        return "A"
    return "tie"


@dataclass
class CasePointwiseOutcome:
    case_id: str
    verdict: PointwiseVerdict | None
    parse_error: bool
    retried: bool
    chars: int
    tokens: int
    input_tokens: int
    output_tokens: int
    cost_usd: float


class JudgeEngine:
    """Unified judge supporting pointwise scoring, bidirectional pairwise A-vs-B, and tracing."""

    def __init__(
        self,
        router: LLMRouter | None = None,
        settings: Settings | None = None,
        audit_log_path: Path | None = None,
        judge_endpoint: ModelEndpointConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.router = router or LLMRouter(settings=self.settings)
        self.judge_endpoint = judge_endpoint or self.settings.p2_judge_config()
        self.audit_log_path = audit_log_path or (self.settings.log_dir / "p2_judge_audit.jsonl")

    def _call_and_parse(
        self,
        run_id: str,
        case_id: str,
        mode: Literal["pointwise", "pairwise"],
        order: Literal["POINTWISE", "AB", "BA"],
        judge_variant: Literal["mitigated", "baseline"],
        messages: list[dict[str, str]],
        full_prompt_text: str,
        model_cls: type[PointwiseVerdict] | type[PairwiseVerdict],
    ) -> tuple[ParseResult, LLMResponse, int, int, float]:
        resp = self.router.complete(
            self.judge_endpoint, messages, json_mode=True, temperature=0.0
        )
        total_in = resp.input_tokens
        total_out = resp.output_tokens
        total_cost = resp.estimated_cost_usd
        total_latency = resp.latency_ms

        def repair_call(repair_prompt: str) -> str:
            nonlocal total_in, total_out, total_cost, total_latency
            repair_msgs = [
                {
                    "role": "system",
                    "content": "Fix JSON formatting/schema errors and output ONLY valid JSON.",
                },
                {"role": "user", "content": repair_prompt},
            ]
            r_resp = self.router.complete(
                self.judge_endpoint, repair_msgs, json_mode=True, temperature=0.0
            )
            total_in += r_resp.input_tokens
            total_out += r_resp.output_tokens
            total_cost += r_resp.estimated_cost_usd
            total_latency += r_resp.latency_ms
            append_jsonl(
                self.audit_log_path,
                AuditTraceRecord(
                    timestamp=utc_now_iso(),
                    run_id=run_id,
                    case_id=f"{case_id}__repair",
                    mode=mode,
                    presentation_order=order,
                    judge_variant=judge_variant,
                    provider_requested=r_resp.provider_requested,
                    provider_used=r_resp.provider_used,
                    model_requested=r_resp.model_requested,
                    model_used=r_resp.model_used,
                    fallback_triggered=r_resp.fallback_triggered,
                    full_prompt=f"[REPAIR]\n{repair_prompt}",
                    raw_response=r_resp.text,
                    parsed_verdict=None,
                    parse_stage="repair_call",
                    retried=True,
                    latency_ms=r_resp.latency_ms,
                    input_tokens=r_resp.input_tokens,
                    output_tokens=r_resp.output_tokens,
                    estimated_cost_usd=r_resp.estimated_cost_usd,
                ),
            )
            return r_resp.text

        parsed = parse_verdict_with_recovery(resp.text, model_cls, repair_fn=repair_call)

        if isinstance(parsed.verdict, PointwiseVerdict):
            score = compute_weighted_score(parsed.verdict)
            parsed.verdict.overall_score = score
            parsed.verdict.passed = (
                score >= self.settings.p2_pass_threshold
                and parsed.verdict.criteria.safety.score >= 3
            )

        append_jsonl(
            self.audit_log_path,
            AuditTraceRecord(
                timestamp=utc_now_iso(),
                run_id=run_id,
                case_id=case_id,
                mode=mode,
                presentation_order=order,
                judge_variant=judge_variant,
                provider_requested=resp.provider_requested,
                provider_used=resp.provider_used,
                model_requested=resp.model_requested,
                model_used=resp.model_used,
                fallback_triggered=resp.fallback_triggered,
                full_prompt=full_prompt_text,
                raw_response=resp.text,
                parsed_verdict=parsed.verdict.model_dump() if parsed.verdict else None,
                parse_stage=parsed.parse_stage,
                retried=parsed.retried,
                latency_ms=total_latency,
                input_tokens=total_in,
                output_tokens=total_out,
                estimated_cost_usd=round(total_cost, 8),
            ),
        )
        return parsed, resp, total_in, total_out, total_cost

    def evaluate_pointwise_case(
        self,
        case: TestCase,
        run_id: str,
        mitigated: bool = True,
    ) -> CasePointwiseOutcome:
        """Score a single TestCase in pointwise mode."""
        rendered = build_pointwise_prompt(case, mitigated=mitigated)
        variant: Literal["mitigated", "baseline"] = "mitigated" if mitigated else "baseline"
        parsed, _, in_tok, out_tok, cost = self._call_and_parse(
            run_id=run_id,
            case_id=case.id,
            mode="pointwise",
            order="POINTWISE",
            judge_variant=variant,
            messages=rendered.as_messages(),
            full_prompt_text=rendered.full_text(),
            model_cls=PointwiseVerdict,
        )
        candidate = case.model_output or ""
        verdict = parsed.verdict if isinstance(parsed.verdict, PointwiseVerdict) else None
        return CasePointwiseOutcome(
            case_id=case.id,
            verdict=verdict,
            parse_error=parsed.parse_error,
            retried=parsed.retried,
            chars=len(candidate),
            tokens=_count_tokens(candidate),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=cost,
        )

    def evaluate_pairwise_case(
        self,
        case: TestCase,
        run_id: str,
        mitigated: bool = True,
        swap_orders: bool = True,
    ) -> tuple[ReconciledPairResult, dict[str, Any]]:
        """Evaluate (A, B) and optionally (B, A) orderings, reconciling flips into ties."""
        variant: Literal["mitigated", "baseline"] = "mitigated" if mitigated else "baseline"
        prompt_ab = build_pairwise_prompt(case, order="AB", mitigated=mitigated)
        p_ab, _, in_ab, out_ab, cost_ab = self._call_and_parse(
            run_id=run_id,
            case_id=case.id,
            mode="pairwise",
            order="AB",
            judge_variant=variant,
            messages=prompt_ab.as_messages(),
            full_prompt_text=prompt_ab.full_text(),
            model_cls=PairwiseVerdict,
        )

        parse_errors = 1 if p_ab.parse_error else 0
        retries = 1 if p_ab.retried else 0
        tot_in, tot_out, tot_cost = in_ab, out_ab, cost_ab

        v_ab = p_ab.verdict if isinstance(p_ab.verdict, PairwiseVerdict) else None
        raw_ab: Literal["A", "B", "tie", "error"] = v_ab.winner if v_ab else "error"

        v_ba: PairwiseVerdict | None = None
        raw_ba_mapped: Literal["A", "B", "tie", "error"] = raw_ab
        if swap_orders:
            prompt_ba = build_pairwise_prompt(case, order="BA", mitigated=mitigated)
            p_ba, _, in_ba, out_ba, cost_ba = self._call_and_parse(
                run_id=run_id,
                case_id=case.id,
                mode="pairwise",
                order="BA",
                judge_variant=variant,
                messages=prompt_ba.as_messages(),
                full_prompt_text=prompt_ba.full_text(),
                model_cls=PairwiseVerdict,
            )
            parse_errors += 1 if p_ba.parse_error else 0
            retries += 1 if p_ba.retried else 0
            tot_in += in_ba
            tot_out += out_ba
            tot_cost += cost_ba
            v_ba = p_ba.verdict if isinstance(p_ba.verdict, PairwiseVerdict) else None
            raw_ba_mapped = _flip_winner(v_ba.winner) if v_ba else "error"

        # Position flip occurs when both orders parsed and selected different outcomes
        flipped = (
            swap_orders
            and raw_ab != "error"
            and raw_ba_mapped != "error"
            and raw_ab != raw_ba_mapped
        )

        if not swap_orders:
            reconciled: Literal["A", "B", "tie"] = "tie" if raw_ab == "error" else raw_ab
        elif raw_ab == raw_ba_mapped and raw_ab in {"A", "B"}:
            reconciled = raw_ab
        else:
            # Require bidirectional agreement to award a win; flipped or errored -> tie
            reconciled = "tie"

        crit_means_a: dict[str, float] = {}
        crit_means_b: dict[str, float] = {}
        for name in CRITERION_NAMES:
            scores_a: list[int] = []
            scores_b: list[int] = []
            if v_ab:
                c_ab = getattr(v_ab.criteria, name)
                scores_a.append(c_ab.score_a)
                scores_b.append(c_ab.score_b)
            if v_ba:
                # In BA order, Candidate A was shown second (score_b) and B first (score_a)
                c_ba = getattr(v_ba.criteria, name)
                scores_a.append(c_ba.score_b)
                scores_b.append(c_ba.score_a)
            crit_means_a[name] = round(statistics.mean(scores_a), 3) if scores_a else 0.0
            crit_means_b[name] = round(statistics.mean(scores_b), 3) if scores_b else 0.0

        mean_a = round(
            sum(crit_means_a[k] * RUBRIC_WEIGHTS[k] for k in CRITERION_NAMES), 4
        )
        mean_b = round(
            sum(crit_means_b[k] * RUBRIC_WEIGHTS[k] for k in CRITERION_NAMES), 4
        )
        regression_b = (
            crit_means_b["safety"] < crit_means_a["safety"]
            or crit_means_b["correctness"] < crit_means_a["correctness"]
        )

        result = ReconciledPairResult(
            case_id=case.id,
            raw_winner_ab=raw_ab,
            raw_winner_ba_mapped=raw_ba_mapped,
            position_flipped=flipped,
            reconciled_winner=reconciled,
            mean_score_a=mean_a,
            mean_score_b=mean_b,
            criterion_means_a=crit_means_a,
            criterion_means_b=crit_means_b,
            chars_a=len(case.output_a or ""),
            chars_b=len(case.output_b or ""),
            tokens_a=_count_tokens(case.output_a or ""),
            tokens_b=_count_tokens(case.output_b or ""),
            safety_or_correctness_regression_b=regression_b,
        )
        meta = {
            "parse_errors": parse_errors,
            "retries": retries,
            "input_tokens": tot_in,
            "output_tokens": tot_out,
            "cost_usd": tot_cost,
        }
        return result, meta

    def run_pointwise_suite(
        self,
        cases: list[TestCase],
        suite_name: str,
        mitigated: bool = True,
        run_id: str | None = None,
    ) -> tuple[SuiteReport, list[CasePointwiseOutcome]]:
        """Run full pointwise evaluation suite and aggregate metrics."""
        rid = run_id or uuid.uuid4().hex[:10]
        outcomes = [self.evaluate_pointwise_case(c, rid, mitigated=mitigated) for c in cases]

        valid = [o for o in outcomes if o.verdict is not None]
        overall_scores = [
            o.verdict.overall_score
            for o in valid
            if o.verdict and o.verdict.overall_score is not None
        ]
        passes = [o.verdict.passed for o in valid if o.verdict and o.verdict.passed is not None]

        crit_means: dict[str, float] = {}
        for name in CRITERION_NAMES:
            vals = [getattr(o.verdict.criteria, name).score for o in valid if o.verdict]
            crit_means[name] = round(statistics.mean(vals), 3) if vals else 0.0

        dist = {str(s): 0 for s in range(1, 6)}
        for o in valid:
            if o.verdict and o.verdict.overall_score is not None:
                bucket = str(min(5, max(1, round(o.verdict.overall_score))))
                dist[bucket] += 1

        report = SuiteReport(
            run_id=rid,
            suite_name=suite_name,
            mode="pointwise",
            total_cases=len(cases),
            pass_rate=round(sum(1 for p in passes if p) / len(passes), 4) if passes else 0.0,
            mean_overall_score=(
                round(statistics.mean(overall_scores), 4) if overall_scores else 0.0
            ),
            criterion_means=crit_means,
            score_std_dev=(
                round(statistics.stdev(overall_scores), 4) if len(overall_scores) > 1 else 0.0
            ),
            score_distribution=dist,
            parse_error_count=sum(1 for o in outcomes if o.parse_error),
            retry_count=sum(1 for o in outcomes if o.retried),
            total_input_tokens=sum(o.input_tokens for o in outcomes),
            total_output_tokens=sum(o.output_tokens for o in outcomes),
            total_cost_usd=round(sum(o.cost_usd for o in outcomes), 6),
        )
        return report, outcomes

    def run_pairwise_suite(
        self,
        cases: list[TestCase],
        suite_name: str,
        mitigated: bool = True,
        swap_orders: bool = True,
        run_id: str | None = None,
    ) -> tuple[SuiteReport, list[ReconciledPairResult]]:
        """Run full pairwise A-vs-B evaluation suite with optional order swap."""
        rid = run_id or uuid.uuid4().hex[:10]
        results: list[ReconciledPairResult] = []
        parse_errors = 0
        retries = 0
        in_tok = 0
        out_tok = 0
        cost = 0.0

        for case in cases:
            res, meta = self.evaluate_pairwise_case(
                case, rid, mitigated=mitigated, swap_orders=swap_orders
            )
            results.append(res)
            parse_errors += meta["parse_errors"]
            retries += meta["retries"]
            in_tok += meta["input_tokens"]
            out_tok += meta["output_tokens"]
            cost += meta["cost_usd"]

        wins_a = sum(1 for r in results if r.reconciled_winner == "A")
        wins_b = sum(1 for r in results if r.reconciled_winner == "B")
        ties = sum(1 for r in results if r.reconciled_winner == "tie")
        flips = sum(1 for r in results if r.position_flipped)

        crit_means_a = {
            name: round(statistics.mean(r.criterion_means_a[name] for r in results), 3)
            if results
            else 0.0
            for name in CRITERION_NAMES
        }
        crit_means_b = {
            name: round(statistics.mean(r.criterion_means_b[name] for r in results), 3)
            if results
            else 0.0
            for name in CRITERION_NAMES
        }

        report = SuiteReport(
            run_id=rid,
            suite_name=suite_name,
            mode="pairwise",
            total_cases=len(cases),
            mean_overall_score=(
                round(statistics.mean(r.mean_score_b for r in results), 4) if results else 0.0
            ),
            criterion_means=crit_means_b,
            wins_a=wins_a,
            wins_b=wins_b,
            ties=ties,
            position_flip_rate=round(flips / len(results), 4) if results else 0.0,
            parse_error_count=parse_errors,
            retry_count=retries,
            total_input_tokens=in_tok,
            total_output_tokens=out_tok,
            total_cost_usd=round(cost, 6),
            extra={"criterion_means_a": crit_means_a, "criterion_means_b": crit_means_b},
        )
        return report, results
