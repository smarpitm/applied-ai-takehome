"""Three-stage malformed JSON verdict parser with Pydantic v2 validation and 1 LLM retry."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@dataclass
class ParseResult:
    """Outcome of parsing raw LLM judge text into a strict Pydantic verdict."""

    verdict: BaseModel | None
    parse_stage: str  # direct | regex_extract | repair_retry | failed
    parse_error: bool
    retried: bool
    error_detail: str | None = None


def extract_outermost_json(raw_text: str) -> str | None:
    """Extract the outermost balanced `{...}` object, stripping fences/thinking tags."""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()

    start = cleaned.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : idx + 1]
                    # Strip trailing commas before closing braces/brackets
                    return re.sub(r",\s*([}\]])", r"\1", candidate)
    return None


def parse_verdict_with_recovery(
    raw_text: str,
    model_cls: type[T],
    repair_fn: Callable[[str], str] | None = None,
) -> ParseResult:
    """Parse structured verdict via 3 stages:
    1. Direct Pydantic `model_validate_json`
    2. Regex outermost JSON extraction + trailing-comma cleanup
    3. Single LLM repair call sending invalid text + exact Pydantic ValidationError
    """
    # Stage 1: Direct strict validation
    try:
        verdict = model_cls.model_validate_json(raw_text.strip())
        return ParseResult(
            verdict=verdict, parse_stage="direct", parse_error=False, retried=False
        )
    except (ValidationError, ValueError) as stage1_err:
        last_err = str(stage1_err)

    # Stage 2: Regex outermost JSON object extraction
    extracted = extract_outermost_json(raw_text)
    if extracted:
        try:
            verdict = model_cls.model_validate_json(extracted)
            return ParseResult(
                verdict=verdict,
                parse_stage="regex_extract",
                parse_error=False,
                retried=False,
            )
        except (ValidationError, ValueError) as stage2_err:
            last_err = str(stage2_err)

    # Stage 3: Single error-guided repair call back to judge
    if repair_fn is not None:
        repair_prompt = (
            "Your previous response failed strict JSON schema validation.\n"
            f"Validation error:\n{last_err}\n\n"
            f"Invalid output:\n{raw_text}\n\n"
            "Return ONLY a single corrected valid JSON object satisfying the schema. "
            "Include quoted evidence and rationale before numeric scores."
        )
        try:
            repaired_raw = repair_fn(repair_prompt)
            try:
                verdict = model_cls.model_validate_json(repaired_raw.strip())
                return ParseResult(
                    verdict=verdict,
                    parse_stage="repair_retry",
                    parse_error=False,
                    retried=True,
                )
            except (ValidationError, ValueError) as stage3_err:
                extracted_retry = extract_outermost_json(repaired_raw)
                if extracted_retry:
                    verdict = model_cls.model_validate_json(extracted_retry)
                    return ParseResult(
                        verdict=verdict,
                        parse_stage="repair_retry",
                        parse_error=False,
                        retried=True,
                    )
                last_err = str(stage3_err)
        except Exception as repair_exc:
            last_err = f"Repair call failed: {repair_exc}"

    return ParseResult(
        verdict=None,
        parse_stage="failed",
        parse_error=True,
        retried=repair_fn is not None,
        error_detail=last_err,
    )
