"""
Optional LLM enhancement layer.

Every stat-producing agent can be annotated with a short, plain-English
interpretation written by an LLM (default: gpt-4o-mini via the OpenAI API).
This module is intentionally the *only* place that talks to the LLM, so
agents stay simple and you can swap providers/models in one spot.

The API key can come from two places, checked in this order for every call:
  1. A per-session runtime key the user typed into the UI and that passed
     validate_api_key() (see main.py's /api/eda/{id}/llm/connect) -- never
     written to disk, .env, or any log line.
  2. The OPENAI_API_KEY environment variable (e.g. from a .env file), as a
     deployment-wide default.

Design rules this module enforces:
  - The LLM only ever *interprets* numbers that are already in the JSON
    payload computed by pandas/scipy/sklearn -- it never invents figures.
  - If no key is available (env or session), is_available() is False and
    every call below becomes a harmless no-op (pipeline runs as normal).
  - Failures are caught, logged as warnings, and never break the pipeline.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional, Tuple

from logger_config import log_warning

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
ENV_API_KEY = os.getenv("OPENAI_API_KEY")
REQUEST_TIMEOUT_SEC = float(os.getenv("LLM_TIMEOUT_SEC", "12"))
MAX_RETRIES = 2


def is_available(session_api_key: Optional[str] = None) -> bool:
    """Whether LLM enhancement can run at all, given an optional session key."""
    return bool(session_api_key or ENV_API_KEY)


def _resolve_key(session_api_key: Optional[str]) -> Optional[str]:
    return session_api_key or ENV_API_KEY


def _build_client(api_key: str):
    from openai import AsyncOpenAI  # imported lazily so the app still runs without the package
    return AsyncOpenAI(api_key=api_key)


def _slim(data: Any, max_chars: int = 3000) -> str:
    """Compact a result payload to a token-friendly JSON string."""
    text = json.dumps(data, default=str)
    return text if len(text) <= max_chars else text[:max_chars] + " …(truncated)"


# --------------------------------------------------------------------------
# Runtime key validation (used by POST /api/eda/{id}/llm/connect)
# --------------------------------------------------------------------------
async def validate_api_key(api_key: str) -> Tuple[bool, str]:
    """
    Validates a user-supplied key with one cheap, real API call.
    Returns (is_valid, message). Never raises.
    """
    if not api_key or not api_key.strip():
        return False, "API key is empty."
    try:
        client = _build_client(api_key.strip())
        await asyncio.wait_for(client.models.list(), timeout=8)
        return True, "Key validated successfully."
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        if "401" in msg or "invalid_api_key" in msg or "Incorrect API key" in msg:
            return False, "That key was rejected by OpenAI (invalid or revoked)."
        if "429" in msg:
            return False, "Key looks valid but is currently rate-limited/out of quota."
        return False, f"Could not verify key: {msg[:160]}"


# --------------------------------------------------------------------------
# Per-step interpretive commentary (steps 1-9)
# --------------------------------------------------------------------------
STEP_SYSTEM_PROMPT = (
    "You are a senior data analyst annotating one step of an automated EDA "
    "pipeline. You will be given structured statistics (JSON) already "
    "computed by pandas/scipy/sklearn for a single analysis step. Write "
    "2-4 concise, plain-English sentences interpreting what these numbers "
    "mean for someone deciding how to clean, transform, or model this "
    "dataset. Only reference figures that literally appear in the JSON -- "
    "never invent or estimate a number that isn't there. Plain prose only: "
    "no markdown, no headers, no bullet points."
)


async def _call_llm(
    system_prompt: str, user_prompt: str, session_id: str, agent_name: str,
    api_key: Optional[str] = None, json_mode: bool = False,
) -> Optional[str]:
    resolved_key = _resolve_key(api_key)
    if not resolved_key:
        return None
    client = _build_client(resolved_key)
    kwargs: Dict[str, Any] = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = await asyncio.wait_for(
                client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=900 if json_mode else 260,
                    temperature=0.4,
                    **kwargs,
                ),
                timeout=REQUEST_TIMEOUT_SEC,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            if attempt < MAX_RETRIES:
                await asyncio.sleep(0.4 * (attempt + 1))
                continue
            log_warning(session_id, agent_name, f"LLM call failed after retries: {exc}")
            return None


async def get_commentary(
    agent_name: str, data: Dict[str, Any], session_id: str, api_key: Optional[str] = None
) -> Optional[str]:
    """Per-step interpretive note. Used by every agent except the final synthesis."""
    prompt = f"EDA step: {agent_name}\nStatistics (JSON):\n{_slim(data)}"
    return await _call_llm(STEP_SYSTEM_PROMPT, prompt, session_id, agent_name, api_key=api_key)


# --------------------------------------------------------------------------
# Business-case report (step 10 -- structured 5-section synthesis)
# --------------------------------------------------------------------------
BUSINESS_CASE_SYSTEM_PROMPT = (
    "You are a senior data analyst delivering the final section of an "
    "automated EDA report for a business case study. You are given: (a) "
    "rule-based insight bullets, (b) short interpretive notes already "
    "written for each individual EDA step, and (c) optional business "
    "context supplied by the user describing what this dataset is for. "
    "Using ONLY the figures and findings present in that material -- never "
    "invent numbers -- write a business-facing report. "
    "Respond with STRICT JSON only (no markdown fences, no prose outside "
    "the JSON) using exactly these keys, each a string containing plain "
    "prose (use \\n\\n between paragraphs if you need more than one, but "
    "no markdown headers or bullet characters):\n"
    "{\n"
    '  "results_and_recommendations": "Present the key results and concrete '
    'recommendations for the business case, with justification tied to '
    'specific findings.",\n'
    '  "industry_implications": "Explore what these results imply for the '
    'industry, business, or policy context.",\n'
    '  "limitations": "Acknowledge limitations or constraints of this study '
    '(data quality, sample size, missing context, methodology caveats) that '
    'may have affected the results.",\n'
    '  "alternative_explanations": "Offer alternative explanations or '
    'recommendations where the data is ambiguous, or state clearly that no '
    'strong alternative applies.",\n'
    '  "key_learnings_methodology": "Summarize the key learnings, the '
    'methodology/tools used (e.g. IQR outlier detection, Shapiro-Wilk '
    'normality test, PCA, K-means), and how this approach applies to the '
    'industry."\n'
    "}"
)


async def get_business_case_report(
    insights: list[str],
    per_step_commentary: Dict[str, str],
    business_context: Optional[str],
    session_id: str,
    api_key: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Structured 5-section business report, used by the Summary agent.
    Returns a dict with the 5 keys described in BUSINESS_CASE_SYSTEM_PROMPT,
    or None if the LLM is unavailable / the call fails / parsing fails.
    """
    prompt = (
        f"Business context (may be empty): {business_context or 'Not provided.'}\n\n"
        f"Rule-based insight bullets:\n{json.dumps(insights)}\n\n"
        f"Per-step interpretive notes:\n{json.dumps(per_step_commentary)}"
    )
    raw = await _call_llm(
        BUSINESS_CASE_SYSTEM_PROMPT, prompt, session_id, "summary",
        api_key=api_key, json_mode=True,
    )
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        log_warning(session_id, "summary", f"business case JSON parse failed: {exc}")
        return None

    expected_keys = [
        "results_and_recommendations", "industry_implications", "limitations",
        "alternative_explanations", "key_learnings_methodology",
    ]
    missing = [k for k in expected_keys if k not in parsed]
    if missing:
        log_warning(session_id, "summary", f"business case JSON missing keys: {missing}")
        return None
    return {k: str(parsed[k]) for k in expected_keys}