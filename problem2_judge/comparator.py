"""A/B comparison runner with binomial sign test, bootstrap CI, and zero-regression gate."""

from __future__ import annotations

import math
import random
import statistics
import uuid
from typing import Any

from problem2_judge.judge_engine import JudgeEngine
from problem2_judge.schemas import TestCase
from shared.config import ModelEndpointConfig, Settings, get_settings
from shared.llm_client import LLMRouter


def binomial_sign_test_p_value(wins_a: int, wins_b: int) -> float:
    """Exact two-sided binomial sign test p-value excluding ties."""
    n = wins_a + wins_b
    if n == 0:
        return 1.0
    k = min(wins_a, wins_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return round(min(1.0, 2.0 * tail), 6)


def bootstrap_mean_diff_ci(
    deltas: list[float],
    n_boot: int = 2000,
    seed: int = 42,
) -> tuple[float, float]:
    """Deterministic 95% bootstrap confidence interval for mean(score_b - score_a)."""
    if not deltas:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(sample))
    means.sort()
    low_idx = max(0, int(0.025 * n_boot))
    high_idx = min(n_boot - 1, int(0.975 * n_boot))
    return (round(means[low_idx], 4), round(means[high_idx], 4))


def generate_ab_candidates(
    cases: list[dict[str, Any]],
    sys_prompt_a: str,
    sys_prompt_b: str,
    router: LLMRouter,
    gen_endpoint: ModelEndpointConfig,
) -> list[TestCase]:
    """Generate fresh Config A and Config B candidate answers via decoupled GeneratorConfig."""
    built: list[TestCase] = []
    for item in cases:
        out_a = router.complete(
            gen_endpoint,
            [
                {"role": "system", "content": sys_prompt_a},
                {"role": "user", "content": item["input"]},
            ],
        ).text
        out_b = router.complete(
            gen_endpoint,
            [
                {"role": "system", "content": sys_prompt_b},
                {"role": "user", "content": item["input"]},
            ],
        ).text
        built.append(
            TestCase(
                id=item["id"],
                category="ab_live",
                input=item["input"],
                system_prompt=sys_prompt_b,
                expected_output=item.get("expected_output"),
                output_a=out_a,
                output_b=out_b,
            )
        )
    return built


def run_ab_comparison(
    ab_suite_data: dict[str, Any],
    engine: JudgeEngine,
    generate_live: bool = False,
    settings: Settings | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run full A/B evaluation (pointwise pass rates + pairwise debiased wins + winner gate)."""
    cfg = settings or get_settings()
    rid = run_id or uuid.uuid4().hex[:8]
    name_a = ab_suite_data["config_a"]["name"]
    name_b = ab_suite_data["config_b"]["name"]
    raw_cases = ab_suite_data["cases"]

    if generate_live:
        cases = generate_ab_candidates(
            raw_cases,
            ab_suite_data["config_a"]["system_prompt"],
            ab_suite_data["config_b"]["system_prompt"],
            router=engine.router,
            gen_endpoint=cfg.p2_generator_config(),
        )
    else:
        cases = [
            TestCase(
                id=c["id"],
                category="ab_cached",
                input=c["input"],
                system_prompt=ab_suite_data["config_b"]["system_prompt"],
                expected_output=c.get("expected_output"),
                output_a=c["output_a"],
                output_b=c["output_b"],
            )
            for c in raw_cases
        ]

    # 1. Pairwise bidirectional evaluation
    pw_report, pair_results = engine.run_pairwise_suite(
        cases,
        suite_name=f"{name_a}_vs_{name_b}",
        mitigated=True,
        swap_orders=True,
        run_id=f"{rid}_ab_pair",
    )

    # 2. Pointwise evaluation for absolute pass rates of Config A and Config B
    cases_a = [
        TestCase(
            id=f"{c.id}__A",
            input=c.input,
            system_prompt=ab_suite_data["config_a"]["system_prompt"],
            expected_output=c.expected_output,
            model_output=c.output_a,
        )
        for c in cases
    ]
    cases_b = [
        TestCase(
            id=f"{c.id}__B",
            input=c.input,
            system_prompt=ab_suite_data["config_b"]["system_prompt"],
            expected_output=c.expected_output,
            model_output=c.output_b,
        )
        for c in cases
    ]
    rep_a, _ = engine.run_pointwise_suite(
        cases_a, suite_name=name_a, mitigated=True, run_id=f"{rid}_pt_a"
    )
    rep_b, _ = engine.run_pointwise_suite(
        cases_b, suite_name=name_b, mitigated=True, run_id=f"{rid}_pt_b"
    )

    # Statistical significance + safety/correctness zero-regression gate
    p_value = binomial_sign_test_p_value(pw_report.wins_a, pw_report.wins_b)
    score_deltas = [r.mean_score_b - r.mean_score_a for r in pair_results]
    ci_low, ci_high = bootstrap_mean_diff_ci(score_deltas)

    safety_regression_b = rep_b.criterion_means["safety"] < rep_a.criterion_means["safety"]
    correctness_regression_b = (
        rep_b.criterion_means["correctness"] < rep_a.criterion_means["correctness"]
    )
    safety_regression_a = rep_a.criterion_means["safety"] < rep_b.criterion_means["safety"]
    correctness_regression_a = (
        rep_a.criterion_means["correctness"] < rep_b.criterion_means["correctness"]
    )

    flip_rate = pw_report.position_flip_rate or 0.0
    stat_sig = p_value < 0.05 or (ci_low > 0.0 or ci_high < 0.0)

    declared_winner = "no_winner_human_review_required"
    reason = "Net win margin did not pass statistical significance or failed guardrails."
    if flip_rate > 0.15:
        declared_winner = "blocked_high_position_flip_rate"
        reason = f"Position flip rate {flip_rate:.1%} exceeds 15% release-gate limit."
    elif stat_sig and pw_report.wins_b > pw_report.wins_a:
        if not safety_regression_b and not correctness_regression_b:
            declared_winner = name_b
            reason = (
                f"{name_b} won {pw_report.wins_b}-{pw_report.wins_a} ({pw_report.ties} ties), "
                f"sign-test p={p_value}, 95% CI [{ci_low}, {ci_high}], "
                "zero safety/correctness regression."
            )
        else:
            reason = f"{name_b} led wins but regressed on safety/correctness."
    elif stat_sig and pw_report.wins_a > pw_report.wins_b:
        if not safety_regression_a and not correctness_regression_a:
            declared_winner = name_a
            reason = (
                f"{name_a} won {pw_report.wins_a}-{pw_report.wins_b} ({pw_report.ties} ties), "
                f"sign-test p={p_value}, 95% CI [{ci_low}, {ci_high}], "
                "zero safety/correctness regression."
            )
        else:
            reason = f"{name_a} led wins but regressed on safety/correctness."

    total_in = pw_report.total_input_tokens + rep_a.total_input_tokens + rep_b.total_input_tokens
    total_out = (
        pw_report.total_output_tokens + rep_a.total_output_tokens + rep_b.total_output_tokens
    )
    total_cost = round(
        pw_report.total_cost_usd + rep_a.total_cost_usd + rep_b.total_cost_usd, 6
    )

    n_cases = len(cases) or 1
    return {
        "run_id": rid,
        "total_cases": len(cases),
        "config_a": {
            "name": name_a,
            "pass_rate": rep_a.pass_rate,
            "mean_score": rep_a.mean_overall_score,
            "win_rate": round(pw_report.wins_a / n_cases, 4),
            "wins": pw_report.wins_a,
            "criterion_means": rep_a.criterion_means,
        },
        "config_b": {
            "name": name_b,
            "pass_rate": rep_b.pass_rate,
            "mean_score": rep_b.mean_overall_score,
            "win_rate": round(pw_report.wins_b / n_cases, 4),
            "wins": pw_report.wins_b,
            "criterion_means": rep_b.criterion_means,
        },
        "ties": pw_report.ties,
        "position_flip_rate": flip_rate,
        "binomial_sign_test_p_value": p_value,
        "bootstrap_95_ci_mean_diff": [ci_low, ci_high],
        "declared_winner": declared_winner,
        "justification": reason,
        "parse_error_count": (
            pw_report.parse_error_count + rep_a.parse_error_count + rep_b.parse_error_count
        ),
        "retry_count": pw_report.retry_count + rep_a.retry_count + rep_b.retry_count,
        "total_judge_tokens": total_in + total_out,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_cost_usd": total_cost,
    }
