"""Judge validation artifact: gold agreement (Cohen's kappa), adversarial probes, 3x retest."""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

from problem2_judge.judge_engine import JudgeEngine
from problem2_judge.schemas import TestCase


def compute_cohens_kappa(y_true: list[Any], y_pred: list[Any]) -> float:
    """Compute exact Cohen's kappa over categorical labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    n = len(y_true)
    if n == 0:
        return 0.0
    p_o = sum(1 for a, b in zip(y_true, y_pred, strict=True) if a == b) / n
    counts_true = Counter(y_true)
    counts_pred = Counter(y_pred)
    labels = set(counts_true) | set(counts_pred)
    p_e = sum((counts_true[lbl] / n) * (counts_pred[lbl] / n) for lbl in labels)
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return round((p_o - p_e) / (1.0 - p_e), 4)


def run_gold_agreement_validation(
    engine: JudgeEngine,
    cases: list[TestCase],
    run_id: str | None = None,
    precomputed_pw_outcomes: list[Any] | None = None,
    precomputed_ab_results: list[Any] | None = None,
) -> dict[str, Any]:
    """Measure raw agreement and Cohen's kappa against hand-labeled gold verdicts."""
    rid = run_id or uuid.uuid4().hex[:8]

    # Pointwise pass/fail agreement
    pw_cases = [c for c in cases if c.gold_pass is not None and c.model_output is not None]
    if precomputed_pw_outcomes is not None and len(precomputed_pw_outcomes) == len(cases):
        pw_outcomes = [
            o
            for c, o in zip(cases, precomputed_pw_outcomes, strict=True)
            if c.gold_pass is not None and c.model_output is not None
        ]
    else:
        _, pw_outcomes = engine.run_pointwise_suite(
            pw_cases, suite_name="gold_pointwise", mitigated=True, run_id=f"{rid}_gold_pw"
        )
    pw_true: list[bool] = []
    pw_pred: list[bool] = []
    for case, out in zip(pw_cases, pw_outcomes, strict=True):
        if (
            out.verdict is not None
            and out.verdict.passed is not None
            and case.gold_pass is not None
        ):
            pw_true.append(case.gold_pass)
            pw_pred.append(out.verdict.passed)

    pw_acc = (
        round(sum(1 for a, b in zip(pw_true, pw_pred, strict=True) if a == b) / len(pw_true), 4)
        if pw_true
        else 0.0
    )
    pw_kappa = compute_cohens_kappa(pw_true, pw_pred)

    # Pairwise winner agreement
    ab_cases = [c for c in cases if c.gold_winner is not None and c.output_a and c.output_b]
    if precomputed_ab_results is not None and len(precomputed_ab_results) == len(cases):
        ab_results = [
            r
            for c, r in zip(cases, precomputed_ab_results, strict=True)
            if c.gold_winner is not None and c.output_a and c.output_b
        ]
    else:
        _, ab_results = engine.run_pairwise_suite(
            ab_cases,
            suite_name="gold_pairwise",
            mitigated=True,
            swap_orders=True,
            run_id=f"{rid}_gold_ab",
        )
    ab_true = [c.gold_winner for c in ab_cases if c.gold_winner is not None]
    ab_pred = [r.reconciled_winner for r in ab_results]
    ab_acc = (
        round(sum(1 for a, b in zip(ab_true, ab_pred, strict=True) if a == b) / len(ab_true), 4)
        if ab_true
        else 0.0
    )
    ab_kappa = compute_cohens_kappa(ab_true, ab_pred)

    return {
        "pointwise_sample_size": len(pw_true),
        "pointwise_agreement_rate": pw_acc,
        "pointwise_cohens_kappa": pw_kappa,
        "pairwise_sample_size": len(ab_true),
        "pairwise_agreement_rate": ab_acc,
        "pairwise_cohens_kappa": ab_kappa,
    }


def run_adversarial_probe_validation(
    engine: JudgeEngine,
    probes: list[TestCase],
    run_id: str | None = None,
    precomputed_base_outcomes: list[Any] | None = None,
    precomputed_mit_outcomes: list[Any] | None = None,
) -> dict[str, Any]:
    """Run all targeted adversarial probes on naive baseline vs mitigated judge."""
    rid = run_id or uuid.uuid4().hex[:8]
    if precomputed_base_outcomes is not None:
        base_outcomes = precomputed_base_outcomes
    else:
        _, base_outcomes = engine.run_pointwise_suite(
            probes, suite_name="adv_baseline", mitigated=False, run_id=f"{rid}_adv_base"
        )
    if precomputed_mit_outcomes is not None:
        mit_outcomes = precomputed_mit_outcomes
    else:
        _, mit_outcomes = engine.run_pointwise_suite(
            probes, suite_name="adv_mitigated", mitigated=True, run_id=f"{rid}_adv_mit"
        )

    def _fooled_stats(outcomes: list[Any]) -> tuple[int, float, dict[str, dict[str, int]]]:
        fooled_total = 0
        by_type: dict[str, dict[str, int]] = {}
        for probe, out in zip(probes, outcomes, strict=True):
            ptype = probe.category
            bucket = by_type.setdefault(ptype, {"total": 0, "fooled": 0})
            bucket["total"] += 1
            pred = out.verdict.passed if out.verdict else None
            # Fooled if judge failed to match expected gold_pass (or errored)
            is_fooled = pred != probe.gold_pass
            if is_fooled:
                fooled_total += 1
                bucket["fooled"] += 1
        rate = round(fooled_total / len(probes), 4) if probes else 0.0
        return fooled_total, rate, by_type

    base_n, base_rate, base_by_type = _fooled_stats(base_outcomes)
    mit_n, mit_rate, mit_by_type = _fooled_stats(mit_outcomes)

    return {
        "total_probes": len(probes),
        "baseline_fooled_count": base_n,
        "baseline_fooled_rate": base_rate,
        "mitigated_fooled_count": mit_n,
        "mitigated_fooled_rate": mit_rate,
        "baseline_by_type": base_by_type,
        "mitigated_by_type": mit_by_type,
    }


def run_test_retest_validation(
    engine: JudgeEngine,
    cases: list[TestCase],
    repeats: int = 3,
    run_id: str | None = None,
    precomputed_run1_outcomes: list[Any] | None = None,
) -> dict[str, Any]:
    """Re-run judge N=3 times at temperature=0.0 and measure verdict consistency."""
    rid = run_id or uuid.uuid4().hex[:8]
    subset = cases[:10]
    runs_verdicts: list[list[bool | None]] = []

    start_idx = 0
    if precomputed_run1_outcomes is not None and len(precomputed_run1_outcomes) >= len(subset):
        run1 = precomputed_run1_outcomes[: len(subset)]
        runs_verdicts.append([o.verdict.passed if o.verdict else None for o in run1])
        start_idx = 1

    for idx in range(start_idx, repeats):
        _, outcomes = engine.run_pointwise_suite(
            subset, suite_name=f"retest_{idx + 1}", mitigated=True, run_id=f"{rid}_rt{idx + 1}"
        )
        runs_verdicts.append([o.verdict.passed if o.verdict else None for o in outcomes])

    flips = 0
    for case_idx in range(len(subset)):
        observed = {runs_verdicts[r][case_idx] for r in range(repeats)}
        if len(observed) > 1:
            flips += 1

    return {
        "sample_size": len(subset),
        "repeats": repeats,
        "temperature": engine.judge_endpoint.temperature,
        "verdict_flips": flips,
        "test_retest_flip_rate": round(flips / len(subset), 4) if subset else 0.0,
    }
