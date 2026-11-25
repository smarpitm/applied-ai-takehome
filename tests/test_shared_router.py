"""Unit tests for shared config, API-first -> local fallback routing, and JSONL logging."""

from pathlib import Path

import pytest

from shared.config import ModelEndpointConfig, Settings
from shared.llm_client import LLMRouter, estimate_cost_usd, infer_model_family
from shared.logging_utils import append_jsonl, read_jsonl


def test_api_first_succeeds_without_fallback() -> None:
    calls: list[tuple[str, str]] = []

    def fake_transport(
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        calls.append((provider, model))
        return '{"ok": true}', 120, 30

    settings = Settings(groq_api_key="gsk_test_dummy", enable_local_fallback=True)
    router = LLMRouter(settings=settings, transport=fake_transport)
    endpoint = ModelEndpointConfig(
        provider="groq",
        model="llama-3.1-8b-instant",
        fallback_provider="ollama",
        fallback_model="llama3.1:latest",
    )

    resp = router.complete(endpoint, [{"role": "user", "content": "hi"}], json_mode=True)
    assert calls == [("groq", "llama-3.1-8b-instant")]
    assert resp.provider_used == "groq"
    assert resp.fallback_triggered is False
    assert resp.estimated_cost_usd > 0.0


def test_api_failure_falls_back_to_local_ollama() -> None:
    calls: list[tuple[str, str]] = []

    def failing_api_then_ollama(
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        calls.append((provider, model))
        if provider == "gemini":
            raise TimeoutError("429 rate limited")
        return "local ok", 90, 20

    settings = Settings(gemini_api_key="gem_test_dummy", enable_local_fallback=True)
    router = LLMRouter(settings=settings, transport=failing_api_then_ollama)
    endpoint = ModelEndpointConfig(
        provider="gemini",
        model="gemini-2.5-flash",
        fallback_provider="ollama",
        fallback_model="qwen3-vl:2b",
    )

    resp = router.complete(endpoint, [{"role": "user", "content": "score"}])
    assert calls == [("gemini", "gemini-2.5-flash"), ("ollama", "qwen3-vl:2b")]
    assert resp.provider_used == "ollama"
    assert resp.model_used == "qwen3-vl:2b"
    assert resp.fallback_triggered is True
    assert "TimeoutError" in (resp.fallback_reason or "")
    assert resp.estimated_cost_usd == 0.0


def test_missing_api_key_without_fallback_raises() -> None:
    settings = Settings(groq_api_key=None, enable_local_fallback=False)
    router = LLMRouter(settings=settings)
    endpoint = settings.p1_generator_config()
    with pytest.raises(RuntimeError, match="local fallback is disabled"):
        router.complete(endpoint, [{"role": "user", "content": "hi"}])


def test_append_jsonl_and_helpers(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.jsonl"
    append_jsonl(log_path, {"case_id": "c1", "score": 4})
    append_jsonl(log_path, {"case_id": "c2", "score": 5})
    rows = read_jsonl(log_path)
    assert len(rows) == 2
    assert rows[1]["case_id"] == "c2"
    assert infer_model_family("llama-3.3-70b-versatile") == "llama"
    assert infer_model_family("gemini-2.5-flash") == "gemini"
    assert estimate_cost_usd("ollama", "llama3.1:latest", 1000, 500) == 0.0
