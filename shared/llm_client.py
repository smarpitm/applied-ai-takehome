"""API-first LLM router with automatic local Ollama fallback and cost estimation."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from shared.config import ModelEndpointConfig, Settings, get_settings

logger = logging.getLogger("applied_ai.llm_client")

# USD per 1M tokens: (input_per_1m, output_per_1m)
MODEL_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.0-flash": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
}

OPENAI_COMPAT_BASE_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}


def infer_model_family(model: str) -> str:
    """Infer coarse model family name for self-enhancement bias detection."""
    lower = model.lower()
    for family in ("llama", "qwen", "gemini", "claude", "gpt", "deepseek", "mistral", "gemma"):
        if family in lower:
            return family
    return lower.split(":")[0].split("-")[0]


def estimate_cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD inference cost ($0.00 for local Ollama)."""
    if provider.lower() == "ollama":
        return 0.0
    lower = model.lower()
    in_rate, out_rate = MODEL_PRICING_USD_PER_1M.get(lower, (0.20, 0.60))
    for key, rates in MODEL_PRICING_USD_PER_1M.items():
        if key in lower:
            in_rate, out_rate = rates
            break
    return round((input_tokens * in_rate + output_tokens * out_rate) / 1_000_000, 8)


@dataclass
class LLMResponse:
    """Normalized response returned by LLMRouter across all providers."""

    text: str
    provider_requested: str
    provider_used: str
    model_requested: str
    model_used: str
    fallback_triggered: bool
    fallback_reason: str | None
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float


TransportFn = Callable[
    [str, str, list[dict[str, str]], float, int, bool],
    tuple[str, int, int],
]


class LLMRouter:
    """Routes completion calls API-first, falling back to local Ollama on error/missing key."""

    def __init__(
        self,
        settings: Settings | None = None,
        transport: TransportFn | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._transport = transport or self._dispatch_provider_call

    def _dispatch_provider_call(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        prov = provider.lower()
        if prov == "anthropic":
            return self._call_anthropic(model, messages, temperature, max_tokens)

        if prov == "ollama":
            base_url = self.settings.ollama_base_url
            api_key = "ollama"
        else:
            api_key = self.settings.get_api_key(prov)
            if not api_key:
                raise RuntimeError(f"Missing API key for provider '{prov}'")
            base_url = OPENAI_COMPAT_BASE_URLS.get(prov)
            if not base_url:
                raise ValueError(f"Unsupported provider '{prov}'")

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=1)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as exc:
            # Some local/compat models reject response_format; retry once without it.
            err_msg = str(exc).lower()
            if json_mode and ("response_format" in err_msg or "json_object" in err_msg):
                kwargs.pop("response_format", None)
                resp = client.chat.completions.create(**kwargs)
            else:
                raise

        content = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
        out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
        return content, in_tok, out_tok

    def _call_anthropic(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int, int]:
        from anthropic import Anthropic

        api_key = self.settings.get_api_key("anthropic")
        if not api_key:
            raise RuntimeError("Missing API key for provider 'anthropic'")

        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_msgs = [m for m in messages if m["role"] != "system"]
        client = Anthropic(api_key=api_key, timeout=60.0, max_retries=1)
        resp = client.messages.create(
            model=model,
            system="\n\n".join(system_parts) if system_parts else "",
            messages=user_msgs,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        usage = getattr(resp, "usage", None)
        in_tok = int(getattr(usage, "input_tokens", 0) or 0)
        out_tok = int(getattr(usage, "output_tokens", 0) or 0)
        return text, in_tok, out_tok

    def complete(
        self,
        endpoint: ModelEndpointConfig,
        messages: list[dict[str, str]],
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> LLMResponse:
        """Execute completion using primary API config, falling back to local Ollama if enabled."""
        temp = endpoint.temperature if temperature is None else temperature
        primary_prov = endpoint.provider.lower()

        # Skip straight to fallback if API key is missing and fallback is allowed.
        should_skip_primary = (
            primary_prov != "ollama"
            and self._transport == self._dispatch_provider_call
            and self.settings.get_api_key(primary_prov) is None
        )

        fallback_reason: str | None = None
        if not should_skip_primary:
            start = time.perf_counter()
            try:
                text, in_tok, out_tok = self._transport(
                    primary_prov,
                    endpoint.model,
                    messages,
                    temp,
                    endpoint.max_tokens,
                    json_mode,
                )
                latency_ms = round((time.perf_counter() - start) * 1000, 2)
                return LLMResponse(
                    text=text,
                    provider_requested=primary_prov,
                    provider_used=primary_prov,
                    model_requested=endpoint.model,
                    model_used=endpoint.model,
                    fallback_triggered=False,
                    fallback_reason=None,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    latency_ms=latency_ms,
                    estimated_cost_usd=estimate_cost_usd(
                        primary_prov, endpoint.model, in_tok, out_tok
                    ),
                )
            except Exception as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
                if not self.settings.enable_local_fallback or primary_prov == "ollama":
                    raise
                logger.warning(
                    "Primary provider %s (%s) failed; falling back to %s (%s): %s",
                    primary_prov,
                    endpoint.model,
                    endpoint.fallback_provider,
                    endpoint.fallback_model,
                    fallback_reason,
                )
        else:
            fallback_reason = f"missing_api_key:{primary_prov}"
            if not self.settings.enable_local_fallback:
                raise RuntimeError(
                    f"Missing API key for {primary_prov} and local fallback is disabled."
                )

        fb_start = time.perf_counter()
        fb_prov = endpoint.fallback_provider.lower()
        text, in_tok, out_tok = self._transport(
            fb_prov,
            endpoint.fallback_model,
            messages,
            temp,
            endpoint.max_tokens,
            json_mode,
        )
        fb_latency_ms = round((time.perf_counter() - fb_start) * 1000, 2)
        return LLMResponse(
            text=text,
            provider_requested=primary_prov,
            provider_used=fb_prov,
            model_requested=endpoint.model,
            model_used=endpoint.fallback_model,
            fallback_triggered=True,
            fallback_reason=fallback_reason,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=fb_latency_ms,
            estimated_cost_usd=estimate_cost_usd(
                fb_prov, endpoint.fallback_model, in_tok, out_tok
            ),
        )
