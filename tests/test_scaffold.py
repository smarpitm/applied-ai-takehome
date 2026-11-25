"""Smoke tests verifying scaffold structure and repository hygiene."""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_required_scaffold_files_exist() -> None:
    expected = [
        ROOT / "README.md",
        ROOT / "pyproject.toml",
        ROOT / ".gitignore",
        ROOT / ".env.example",
        ROOT / "docker-compose.yml",
        ROOT / "docs" / "Gen AI_assignment.pdf",
        ROOT / "docs" / "submission_template (1).docx",
        ROOT / "shared" / "__init__.py",
        ROOT / "problem1_rag" / "__init__.py",
        ROOT / "problem2_judge" / "__init__.py",
    ]
    missing = [str(p) for p in expected if not p.exists()]
    assert not missing, f"Missing scaffold files: {missing}"


def test_pyproject_metadata_valid() -> None:
    with (ROOT / "pyproject.toml").open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["name"] == "applied-ai-takehome"
    assert ">=3.11" in data["project"]["requires-python"]


def test_env_example_has_no_secret_values() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if "API_KEY" in key or "SECRET" in key or "TOKEN" in key:
            if "TOKENS" not in key:
                assert value == "", f"Secret placeholder {key} must be empty in .env.example"
