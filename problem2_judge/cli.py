"""CLI entrypoint for Problem 2 Bias-Aware LLM-as-Judge Evaluation Pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
from rich.console import Console
from rich.table import Table

from problem2_judge.bias_checks import run_all_bias_experiments
from problem2_judge.comparator import run_ab_comparison
from problem2_judge.judge_engine import JudgeEngine
from problem2_judge.schemas import TestCase
from problem2_judge.validation import (
    run_adversarial_probe_validation,
    run_gold_agreement_validation,
    run_test_retest_validation,
)
from shared.config import get_settings

app = typer.Typer(help="Problem 2: Bias-Aware LLM-as-Judge Evaluation Pipeline CLI")
console = Console()


def _load_yaml_or_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(raw)
    return yaml.safe_load(raw)


def _load_cases(path: Path) -> tuple[str, list[TestCase]]:
    data = _load_yaml_or_json(path)
    items = data.get("cases") or data.get("probes") or []
    cases: list[TestCase] = []
    for item in items:
        item_copy = dict(item)
        if "probe_type" in item_copy and "category" not in item_copy:
            item_copy["category"] = item_copy.pop("probe_type")
        cases.append(TestCase.model_validate(item_copy))
    return str(data.get("suite_name", path.stem)), cases


@app.command("run-suite")
def run_suite(
    suite_path: Annotated[Path, typer.Option()] = Path("problem2_judge/data/suite_main.yaml"),
    mode: Annotated[str, typer.Option(help="pointwise or pairwise")] = "pairwise",
    mitigated: Annotated[bool, typer.Option()] = True,
    out_path: Annotated[Path, typer.Option()] = Path("results/p2_suite_report.json"),
) -> None:
    """Run pointwise or pairwise evaluation on a JSON/YAML test suite."""
    suite_name, cases = _load_cases(suite_path)
    engine = JudgeEngine()
    if mode.lower() == "pointwise":
        report, _ = engine.run_pointwise_suite(cases, suite_name=suite_name, mitigated=mitigated)
    else:
        report, _ = engine.run_pairwise_suite(
            cases, suite_name=suite_name, mitigated=mitigated, swap_orders=True
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print_json(report.model_dump_json())


@app.command("ab-compare")
def ab_compare(
    ab_path: Annotated[
        Path, typer.Option()
    ] = Path("problem2_judge/data/ab_configs/ab_suite.yaml"),
    generate_live: Annotated[
        bool, typer.Option(help="Generate fresh A/B outputs via GeneratorConfig")
    ] = False,
    out_path: Annotated[Path, typer.Option()] = Path("results/p2_ab_report.json"),
) -> None:
    """Compare Config A vs Config B and apply statistical + safety/correctness winner gate."""
    data = _load_yaml_or_json(ab_path)
    engine = JudgeEngine()
    report = run_ab_comparison(data, engine=engine, generate_live=generate_live)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    console.print_json(json.dumps(report))


@app.command("bias-audit")
def bias_audit(
    main_path: Annotated[Path, typer.Option()] = Path("problem2_judge/data/suite_main.yaml"),
    verbosity_path: Annotated[
        Path, typer.Option()
    ] = Path("problem2_judge/data/padded_verbosity.yaml"),
    probes_path: Annotated[
        Path, typer.Option()
    ] = Path("problem2_judge/data/adversarial_probes.yaml"),
    out_path: Annotated[Path, typer.Option()] = Path("results/p2_bias_audit.json"),
) -> None:
    """Run controlled before-vs-after experiments across all 5 bias categories."""
    _, main_cases = _load_cases(main_path)
    verb_data = _load_yaml_or_json(verbosity_path)
    _, probe_cases = _load_cases(probes_path)
    report = run_all_bias_experiments(
        main_cases=main_cases,
        verbosity_pairs=verb_data["pairs"],
        probes=probe_cases,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    table = Table(title="Problem 2 — Controlled Bias Audit (Before -> After)")
    table.add_column("Bias")
    table.add_column("Metric")
    table.add_column("Before")
    table.add_column("After")
    for row in report["summary_table"]:
        table.add_row(row["bias"], row["metric_name"], row["before_value"], row["after_value"])
    console.print(table)


@app.command("validate")
def validate_judge(
    main_path: Annotated[Path, typer.Option()] = Path("problem2_judge/data/suite_main.yaml"),
    probes_path: Annotated[
        Path, typer.Option()
    ] = Path("problem2_judge/data/adversarial_probes.yaml"),
    out_path: Annotated[Path, typer.Option()] = Path("results/p2_validation_report.json"),
) -> None:
    """Run gold agreement (Cohen's kappa), adversarial probes, and 3x test-retest."""
    _, main_cases = _load_cases(main_path)
    _, probe_cases = _load_cases(probes_path)
    engine = JudgeEngine()

    gold = run_gold_agreement_validation(engine, main_cases)
    adv = run_adversarial_probe_validation(engine, probe_cases)
    retest = run_test_retest_validation(engine, main_cases, repeats=3)

    payload = {
        "gold_agreement": gold,
        "adversarial_probes": adv,
        "test_retest": retest,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print_json(json.dumps(payload))


@app.command("run-all")
def run_all(
    out_dir: Annotated[Path, typer.Option()] = Path("results"),
    generate_live_ab: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run validation, all 5 bias audits, and A/B comparison, saving artifacts to results/."""
    settings = get_settings()
    _, main_cases = _load_cases(Path("problem2_judge/data/suite_main.yaml"))
    verb_data = _load_yaml_or_json(Path("problem2_judge/data/padded_verbosity.yaml"))
    _, probe_cases = _load_cases(Path("problem2_judge/data/adversarial_probes.yaml"))
    ab_data = _load_yaml_or_json(Path("problem2_judge/data/ab_configs/ab_suite.yaml"))

    engine = JudgeEngine(settings=settings)
    bias_rep = run_all_bias_experiments(
        main_cases=main_cases,
        verbosity_pairs=verb_data["pairs"],
        probes=probe_cases,
        settings=settings,
    )
    gold = run_gold_agreement_validation(engine, main_cases)
    adv = run_adversarial_probe_validation(engine, probe_cases)
    retest = run_test_retest_validation(engine, main_cases, repeats=3)
    ab_rep = run_ab_comparison(
        ab_data, engine=engine, generate_live=generate_live_ab, settings=settings
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "p2_bias_audit.json").write_text(json.dumps(bias_rep, indent=2), encoding="utf-8")
    (out_dir / "p2_validation_report.json").write_text(
        json.dumps(
            {"gold_agreement": gold, "adversarial_probes": adv, "test_retest": retest},
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "p2_ab_report.json").write_text(json.dumps(ab_rep, indent=2), encoding="utf-8")
    console.print("Saved p2_bias_audit.json, p2_validation_report.json, and p2_ab_report.json")


if __name__ == "__main__":
    app()
