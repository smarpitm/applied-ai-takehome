"""Controlled before-and-after experiments across all five LLM-as-Judge bias categories."""

from __future__ import annotations

import statistics
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from problem2_judge.judge_engine import JudgeEngine
from problem2_judge.schemas import TestCase
from shared.config import ModelEndpointConfig, Settings, get_settings
from shared.llm_client import LLMRouter, infer_model_family


@dataclass
class BiasSummaryRow:
    """Measured before -> after metrics for one bias category."""

    bias: str
    mitigation: str
    metric_name: str
    before_value: str
    after_value: str
    delta_note: str


def run_position_bias_experiment(
    engine: JudgeEngine,
    cases: list[TestCase],
    run_id: str,
    mit_pairwise: tuple[Any, list[Any]] | None = None,
) -> dict[str, Any]:
    """Compare raw AB/BA flip rate before reconciliation vs reconciled tie handling."""
    base_report, _ = engine.run_pairwise_suite(
        cases,
        suite_name="position_baseline",
        mitigated=False,
        swap_orders=True,
        run_id=f"{run_id}_pos_base",
    )
    if mit_pairwise is not None:
        mit_report, mit_results = mit_pairwise
    else:
        mit_report, mit_results = engine.run_pairwise_suite(
            cases,
            suite_name="position_mitigated",
            mitigated=True,
            swap_orders=True,
            run_id=f"{run_id}_pos_mit",
        )
    inconsistent_ties = sum(1 for r in mit_results if r.position_flipped)
    return {
        "baseline_raw_flip_rate": base_report.position_flip_rate,
        "mitigated_raw_flip_rate": mit_report.position_flip_rate,
        "mitigated_reconciled_wins_from_flips": 0,
        "flipped_pairs_converted_to_tie": inconsistent_ties,
        "total_pairs": len(cases),
    }


def run_verbosity_bias_experiment(
    engine: JudgeEngine,
    verbosity_pairs: list[dict[str, Any]],
    run_id: str,
) -> dict[str, Any]:
    """Compare how often naive vs mitigated judge prefers padded over concise answers."""
    cases = [
        TestCase(
            id=p["id"],
            category="verbosity_probe",
            input=p["input"],
            system_prompt=p.get("system_prompt", ""),
            expected_output=p.get("expected_output"),
            output_a=p["concise_accurate"],
            output_b=p["padded_accurate"],
        )
        for p in verbosity_pairs
    ]

    _, base_results = engine.run_pairwise_suite(
        cases,
        suite_name="verbosity_baseline",
        mitigated=False,
        swap_orders=True,
        run_id=f"{run_id}_verb_base",
    )
    _, mit_results = engine.run_pairwise_suite(
        cases,
        suite_name="verbosity_mitigated",
        mitigated=True,
        swap_orders=True,
        run_id=f"{run_id}_verb_mit",
    )

    n = len(cases) or 1
    base_padded_wins = sum(1 for r in base_results if r.reconciled_winner == "B")
    mit_padded_wins = sum(1 for r in mit_results if r.reconciled_winner == "B")
    mit_concise_wins = sum(1 for r in mit_results if r.reconciled_winner == "A")
    avg_chars_concise = round(statistics.mean(r.chars_a for r in mit_results), 1)
    avg_chars_padded = round(statistics.mean(r.chars_b for r in mit_results), 1)
    avg_tokens_concise = round(statistics.mean(r.tokens_a for r in mit_results), 1)
    avg_tokens_padded = round(statistics.mean(r.tokens_b for r in mit_results), 1)

    return {
        "total_pairs": len(cases),
        "avg_chars_concise": avg_chars_concise,
        "avg_chars_padded": avg_chars_padded,
        "avg_tokens_concise": avg_tokens_concise,
        "avg_tokens_padded": avg_tokens_padded,
        "baseline_padded_win_rate": round(base_padded_wins / n, 4),
        "mitigated_padded_win_rate": round(mit_padded_wins / n, 4),
        "mitigated_concise_win_rate": round(mit_concise_wins / n, 4),
        "mitigated_conciseness_concise": round(
            statistics.mean(r.criterion_means_a["conciseness"] for r in mit_results), 3
        ),
        "mitigated_conciseness_padded": round(
            statistics.mean(r.criterion_means_b["conciseness"] for r in mit_results), 3
        ),
    }


def run_self_enhancement_experiment(
    router: LLMRouter,
    settings: Settings,
    cases: list[TestCase],
    audit_log_path: Path,
    run_id: str,
    precomputed_cross_results: list[Any] | None = None,
) -> dict[str, Any]:
    """Compare same-family judge vs cross-family judge win-rate on Llama vs non-Llama pairs."""
    gen_cfg = settings.p2_generator_config()
    judge_cfg = settings.p2_judge_config()

    same_family_endpoint = ModelEndpointConfig(
        provider=gen_cfg.provider,
        model=gen_cfg.model,
        temperature=0.0,
        max_tokens=1536,
        fallback_provider="ollama",
        fallback_model=settings.ollama_p2_generator_fallback_model,
    )
    same_engine = JudgeEngine(
        router=router,
        settings=settings,
        audit_log_path=audit_log_path,
        judge_endpoint=same_family_endpoint,
    )
    cross_engine = JudgeEngine(
        router=router,
        settings=settings,
        audit_log_path=audit_log_path,
        judge_endpoint=judge_cfg,
    )

    subset = cases[:8]
    same_rep, _ = same_engine.run_pairwise_suite(
        subset,
        suite_name="self_enh_same",
        mitigated=True,
        swap_orders=True,
        run_id=f"{run_id}_se_same",
    )
    if precomputed_cross_results is not None and len(precomputed_cross_results) >= len(subset):
        cross_slice = precomputed_cross_results[: len(subset)]
        cross_wins_a = sum(1 for r in cross_slice if r.reconciled_winner == "A")
        cross_mean_b = round(statistics.mean(r.mean_score_b for r in cross_slice), 4)
    else:
        cross_rep, _ = cross_engine.run_pairwise_suite(
            subset,
            suite_name="self_enh_cross",
            mitigated=True,
            swap_orders=True,
            run_id=f"{run_id}_se_cross",
        )
        cross_wins_a = cross_rep.wins_a
        cross_mean_b = cross_rep.mean_overall_score
    n = len(subset) or 1
    active_gen_model = (
        gen_cfg.fallback_model
        if not settings.get_api_key(gen_cfg.provider)
        else gen_cfg.model
    )
    return {
        "generator_family": infer_model_family(active_gen_model),
        "same_family_judge_model": same_family_endpoint.model,
        "cross_family_judge_model": judge_cfg.model,
        "cross_family_fallback_model": judge_cfg.fallback_model,
        "same_family_win_rate_a": round(same_rep.wins_a / n, 4),
        "cross_family_win_rate_a": round(cross_wins_a / n, 4),
        "same_family_mean_score_b": same_rep.mean_overall_score,
        "cross_family_mean_score_b": cross_mean_b,
    }


def run_sycophancy_experiment(
    engine: JudgeEngine,
    probes: list[TestCase],
    run_id: str,
    precomputed_base_out: list[Any] | None = None,
    precomputed_mit_out: list[Any] | None = None,
) -> dict[str, Any]:
    """Measure false-pass rate on confidently-wrong / sycophancy probes before vs after."""
    wrong_cats = {
        "confident_hallucination",
        "verbose_but_wrong",
        "polite_instruction_violation",
    }
    if precomputed_base_out is not None and precomputed_mit_out is not None:
        base_out = [
            o for p, o in zip(probes, precomputed_base_out, strict=True) if p.category in wrong_cats
        ]
        mit_out = [
            o for p, o in zip(probes, precomputed_mit_out, strict=True) if p.category in wrong_cats
        ]
        syco_probes = [p for p in probes if p.category in wrong_cats]
    else:
        syco_probes = [p for p in probes if p.category in wrong_cats]
        _, base_out = engine.run_pointwise_suite(
            syco_probes,
            suite_name="sycophancy_baseline",
            mitigated=False,
            run_id=f"{run_id}_syco_base",
        )
        _, mit_out = engine.run_pointwise_suite(
            syco_probes,
            suite_name="sycophancy_mitigated",
            mitigated=True,
            run_id=f"{run_id}_syco_mit",
        )

    n = len(syco_probes) or 1
    base_fp = sum(1 for o in base_out if o.verdict and o.verdict.passed)
    mit_fp = sum(1 for o in mit_out if o.verdict and o.verdict.passed)
    return {
        "probe_count": len(syco_probes),
        "baseline_false_pass_rate": round(base_fp / n, 4),
        "mitigated_false_pass_rate": round(mit_fp / n, 4),
    }


def run_score_clustering_experiment(
    engine: JudgeEngine,
    cases: list[TestCase],
    run_id: str,
    mit_pointwise_report: Any | None = None,
) -> dict[str, Any]:
    """Measure pointwise score std-dev and 1-5 spread before vs after 1/3/5 anchors."""
    base_rep, _ = engine.run_pointwise_suite(
        cases, suite_name="clustering_baseline", mitigated=False, run_id=f"{run_id}_clust_base"
    )
    if mit_pointwise_report is not None:
        mit_rep = mit_pointwise_report
    else:
        mit_rep, _ = engine.run_pointwise_suite(
            cases, suite_name="clustering_mitigated", mitigated=True, run_id=f"{run_id}_clust_mit"
        )
    return {
        "total_cases": len(cases),
        "baseline_score_std_dev": base_rep.score_std_dev,
        "mitigated_score_std_dev": mit_rep.score_std_dev,
        "baseline_distribution": base_rep.score_distribution,
        "mitigated_distribution": mit_rep.score_distribution,
    }


def run_all_bias_experiments(
    main_cases: list[TestCase],
    verbosity_pairs: list[dict[str, Any]],
    probes: list[TestCase],
    router: LLMRouter | None = None,
    settings: Settings | None = None,
    audit_log_path: Path | None = None,
    mit_pairwise: tuple[Any, list[Any]] | None = None,
    mit_pointwise: tuple[Any, list[Any]] | None = None,
    adv_base_outcomes: list[Any] | None = None,
    adv_mit_outcomes: list[Any] | None = None,
) -> dict[str, Any]:
    """Execute controlled before-vs-after experiments across all 5 bias categories."""
    cfg = settings or get_settings()
    llm = router or LLMRouter(settings=cfg)
    log_path = audit_log_path or (cfg.log_dir / "p2_judge_audit.jsonl")
    engine = JudgeEngine(router=llm, settings=cfg, audit_log_path=log_path)
    run_id = uuid.uuid4().hex[:8]

    pos = run_position_bias_experiment(engine, main_cases, run_id, mit_pairwise=mit_pairwise)
    verb = run_verbosity_bias_experiment(engine, verbosity_pairs, run_id)
    se = run_self_enhancement_experiment(
        llm,
        cfg,
        main_cases,
        log_path,
        run_id,
        precomputed_cross_results=mit_pairwise[1] if mit_pairwise else None,
    )
    syco = run_sycophancy_experiment(
        engine,
        probes,
        run_id,
        precomputed_base_out=adv_base_outcomes,
        precomputed_mit_out=adv_mit_outcomes,
    )
    clust = run_score_clustering_experiment(
        engine,
        main_cases,
        run_id,
        mit_pointwise_report=mit_pointwise[0] if mit_pointwise else None,
    )

    summary_rows = [
        BiasSummaryRow(
            bias="Position (A/B order)",
            mitigation="Bidirectional AB + BA evaluation; flipped pairs reconciled to tie",
            metric_name="Flip rate / unverified win leaks",
            before_value=f"{(pos['baseline_raw_flip_rate'] or 0.0) * 100:.1f}%",
            after_value=(
                f"{(pos['mitigated_raw_flip_rate'] or 0.0) * 100:.1f}% raw / "
                "0% unverified wins"
            ),
            delta_note=f"{pos['flipped_pairs_converted_to_tie']} flipped pair(s) downgraded to tie",
        ),
        BiasSummaryRow(
            bias="Verbosity / length",
            mitigation="Conciseness length-penalty anchor + anti-padding instruction",
            metric_name="Padded-over-concise win rate",
            before_value=f"{verb['baseline_padded_win_rate'] * 100:.1f}%",
            after_value=f"{verb['mitigated_padded_win_rate'] * 100:.1f}%",
            delta_note=(
                f"Conciseness: concise={verb['mitigated_conciseness_concise']}, "
                f"padded={verb['mitigated_conciseness_padded']}"
            ),
        ),
        BiasSummaryRow(
            bias="Self-enhancement",
            mitigation="Decoupled cross-family judge vs generator routing",
            metric_name="Candidate A win rate (same vs cross family)",
            before_value=f"{se['same_family_win_rate_a'] * 100:.1f}%",
            after_value=f"{se['cross_family_win_rate_a'] * 100:.1f}%",
            delta_note=(
                f"Gen family={se['generator_family']} vs "
                f"judge={se['cross_family_judge_model']}"
            ),
        ),
        BiasSummaryRow(
            bias="Sycophancy / style",
            mitigation="Evidence-first schema (quotes + rationale before score) + factual check",
            metric_name="False-pass rate on wrong polished probes",
            before_value=f"{syco['baseline_false_pass_rate'] * 100:.1f}%",
            after_value=f"{syco['mitigated_false_pass_rate'] * 100:.1f}%",
            delta_note=f"Tested on {syco['probe_count']} adversarial wrong probes",
        ),
        BiasSummaryRow(
            bias="Score clustering",
            mitigation="Concrete 1/3/5 rubric anchors + pairwise tie-breaking",
            metric_name="Pointwise score std-dev",
            before_value=f"{clust['baseline_score_std_dev']:.3f}",
            after_value=f"{clust['mitigated_score_std_dev']:.3f}",
            delta_note=(
                f"Dist shifted {clust['baseline_distribution']} -> "
                f"{clust['mitigated_distribution']}"
            ),
        ),
    ]

    return {
        "run_id": run_id,
        "summary_table": [asdict(r) for r in summary_rows],
        "details": {
            "position": pos,
            "verbosity": verb,
            "self_enhancement": se,
            "sycophancy": syco,
            "score_clustering": clust,
        },
    }
