import logging

from google.genai import types
import os
from dotenv import load_dotenv
import time
import uuid
import random

load_dotenv()

gemini_models = ["gemini-2.5-pro", "gemini-2.5-flash"]


# ── Gemini helper ──────────────────────────────────────────────────────
import json, re, os, asyncio
from google import genai

# compile once, reuse from the original file
BOUNDARY_JSON_RE = re.compile(
    r"\[\s*\[\s*\d+\s*,\s*\d+\s*](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*])*\s*]"
)

# Configure the client (do this exactly once in your program)
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])


# Suggested defaults that are permissive but still within provider policy
from google.genai.types import HarmCategory, HarmBlockThreshold

import asyncio
import random
from typing import Callable, Optional, Union, List

from google.genai import types as gatypes
# If you previously used `types` directly, I alias it to `gatypes` here for clarity.

# ── Build a relaxed (policy-compliant) safety list using only enums present in your SDK ──
def _build_relaxed_safety() -> List[gatypes.SafetySetting]:
    categories = [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    ]
    out = []
    for name in categories:
        if hasattr(gatypes.HarmCategory, name):
            out.append(
                gatypes.SafetySetting(
                    category=getattr(gatypes.HarmCategory, name),
                    threshold=gatypes.HarmBlockThreshold.BLOCK_NONE,
                )
            )
    return out

RELAXED_SAFETY = _build_relaxed_safety()

# ── YOUR requested function (integrated as-is, referencing RELAXED_SAFETY) ──
def _build_config(
    base: Optional[dict],
    safety_mode: str
) -> gatypes.GenerateContentConfig:
    """
    Build a GenerateContentConfig that:
      • Uses valid HarmCategory enums (no warnings)
      • Disables any tool use / external grounding (no web/search/functions)
    """
    base = dict(base or {})

    # 1) Safety (relaxed means 'BLOCK_ONLY_HIGH' across categories)
    if safety_mode == "relaxed":
        base.setdefault("safety_settings", RELAXED_SAFETY)

    # 2) Absolutely NO external tools / grounding
    #    - tools: []
    #    - tool_config: None
    #    - grounding: None (if present in your SDK version)
    base["tools"] = []                 # prevent function/tool use
    base["tool_config"] = None
    # base["grounding"] = None           # ensure no google search grounding (if supported)

    return gatypes.GenerateContentConfig(**base)

# ── Helpers ──
def _extract_text_and_reason(resp):
    text = getattr(resp, "text", None)
    finish_reason = None
    safety = None
    prompt_feedback = getattr(resp, "prompt_feedback", None)

    try:
        cands = getattr(resp, "candidates", None) or []
        if cands:
            finish_reason = getattr(cands[0], "finish_reason", None)
            safety = getattr(cands[0], "safety_ratings", None)
    except Exception:
        pass

    if not finish_reason and hasattr(resp, "finish_reason"):
        finish_reason = getattr(resp, "finish_reason")

    try:
        if prompt_feedback and hasattr(prompt_feedback, "block_reason") and not finish_reason:
            finish_reason = prompt_feedback.block_reason
    except Exception:
        pass

    return text, finish_reason, safety, prompt_feedback

# ── Main call ──
async def call_gemini_async(
    prompt: str,
    model: str = "gemini-2.5-pro",
    generation_config: Optional[dict] = None,
    return_text_only: bool = False,
    valid_func: Optional[Callable[[Union[str, object]], bool]] = None,
    allow_forced_return: bool = False,
    safety_mode: str = "relaxed",      # "default" | "relaxed"
    max_try: int = 5,
    base_backoff_sec: float = 5.0,
    backoff_jitter_sec: float = 0.5,
    add_unique_id: bool = True,
):
    """
    Gemini call that:
      • Uses your integrated _build_config (no tools/search; relaxed safety optional)
      • Handles safety blocks gracefully
      • Async exponential backoff with jitter
    """
    config = _build_config(generation_config, safety_mode=safety_mode)

    attempts_left = max_try
    last_response = None
    last_error = None

    if add_unique_id:
        prompt = f"Request id:{uuid.uuid4()}\n\n" + prompt

    while attempts_left > 0:
        attempts_left -= 1
        try:
            resp = await gemini_client.aio.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            if resp.text != None:
                last_response = resp

            text, finish_reason, safety, prompt_feedback = _extract_text_and_reason(resp)

            # Safety-limited output: return structured diagnostics rather than failing.
            if str(finish_reason).upper() in {"SAFETY", "BLOCKED", "OTHER"} or (
                prompt_feedback and getattr(prompt_feedback, "block_reason", None)
            ):
                result = {
                    "text": text or "",
                    "finish_reason": finish_reason,
                    "safety_ratings": safety,
                    "prompt_feedback": prompt_feedback,
                    "note": "Generation was limited by safety filters."
                }
                return result["text"] if return_text_only else result

            # Optional caller validation
            if valid_func:
                try:
                    is_valid = valid_func(text)
                except Exception:
                    is_valid = False

                if not is_valid:
                    logging.info("Invalid response, attempt left: {}".format(attempts_left))
                    delay = base_backoff_sec * (2 ** (max_try - attempts_left - 1)) + random.uniform(0, backoff_jitter_sec)
                    await asyncio.sleep(delay)
                    continue

            # Success
            return text if return_text_only else resp

        except Exception as e:
            if attempts_left > 0:
                delay = base_backoff_sec * (2 ** (max_try - attempts_left - 1)) + random.uniform(0, backoff_jitter_sec)
                logging.info( f"[Gemini] error: {e}. Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)

    # Out of retries
    if allow_forced_return and last_response is not None:
        logging.info( f"Warning: forced return after {max_try} attempts.")
        t, _, _, _ = _extract_text_and_reason(last_response)
        return t if return_text_only else last_response

    return None


def call_gemini_sync(prompt: str, model=gemini_models[0],
    generation_config: dict | None = None, return_text_only: bool = False, valid_func= None) -> str | None:
    complete = False
    try_left = 5
    config = None

    if generation_config:
        config = types.GenerateContentConfig(**generation_config)

    while not complete and try_left > 0:
        try:
            try_left -= 1
            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=config
            )
            return response.text
        except Exception as e:
            print(e)
            time.sleep(30)


def is_rate_limited(err) -> bool:
    """Return True if the exception represents a 429/quota hit."""
    # Try structured attributes first
    code = getattr(err, "code", None)
    status = getattr(err, "status", None)
    if code == 429 or (isinstance(status, str) and "RESOURCE_EXHAUSTED" in status):
        return True
    # Fallback: parse stringified message/body
    s = str(err)
    return "RESOURCE_EXHAUSTED" in s or '"code": 429' in s or " 429 " in s