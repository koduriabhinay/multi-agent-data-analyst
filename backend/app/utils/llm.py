"""
LLM client — thin wrapper over Anthropic / OpenAI.

Two things matter here:
1. Retry with backoff, because rate limits happen.
2. `ask_json` reliably returns a dict, because LLMs wrap JSON in prose
   and markdown fences no matter how firmly you ask them not to.

If no API key is set, the client runs in OFFLINE mode and returns
deterministic stub responses so you can develop the pipeline without
burning tokens.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

log = logging.getLogger(__name__)

MAX_RETRIES = 3


class LLMError(RuntimeError):
    """Raised when the model can't be reached or returns unusable output."""


class LLMClient:
    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> None:
        self.provider = provider or os.getenv("LLM_PROVIDER", "anthropic")
        self.model = model or os.getenv("LLM_MODEL", "claude-sonnet-4-6")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self.offline = False

        try:
            self._client = self._build_client()
        except LLMError as exc:
            log.warning("LLM offline (%s). Using stub responses.", exc)
            self.offline = True

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _build_client(self):
        if self.provider == "anthropic":
            key = os.getenv("ANTHROPIC_API_KEY")
            if not key:
                raise LLMError("ANTHROPIC_API_KEY not set")
            from anthropic import Anthropic

            return Anthropic(api_key=key)

        if self.provider == "openai":
            key = os.getenv("OPENAI_API_KEY")
            if not key:
                raise LLMError("OPENAI_API_KEY not set")
            from openai import OpenAI

            return OpenAI(api_key=key)

        raise LLMError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, prompt: str, system: str = "") -> str:
        """Send a prompt, get raw text back."""
        if self.offline:
            return _stub_response(prompt)

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                return self._call(prompt, system)
            except Exception as exc:
                last_error = exc

                # No amount of waiting fixes an exhausted quota or a bad key.
                # Retrying just multiplies the delay before the caller can fall
                # back, so stop now and switch to offline for the rest of the run.
                if _is_permanent(exc):
                    reason = _explain(exc)
                    log.error("Language model unavailable: %s", reason)
                    self.offline = True
                    raise LLMError(reason) from exc

                wait = 2**attempt
                log.warning(
                    "LLM call failed (attempt %d of %d), retrying in %ds: %s",
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                    exc,
                )
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

        raise LLMError(f"LLM failed after {MAX_RETRIES} attempts: {last_error}")

    def ask_json(
        self, prompt: str, system: str = "", fallback: dict | None = None
    ) -> dict[str, Any]:
        """Send a prompt and parse the reply as JSON.

        Falls back to `fallback` rather than crashing the pipeline, because a
        malformed plan shouldn't kill an analysis that can still run on defaults.
        """
        system = system or "Reply with valid JSON only. No prose, no markdown fences."
        try:
            raw = self.ask(prompt, system)
            return extract_json(raw)
        except (LLMError, ValueError) as exc:
            log.warning("JSON parse failed, using fallback: %s", exc)
            if fallback is None:
                raise
            return fallback

    # ------------------------------------------------------------------
    # Provider-specific calls
    # ------------------------------------------------------------------
    def _call(self, prompt: str, system: str) -> str:
        if self.provider == "anthropic":
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system or "You are a precise data analyst.",
                messages=[{"role": "user", "content": prompt}],
            )
            return "".join(block.text for block in resp.content if block.type == "text")

        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system or "You are a precise data analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _is_permanent(exc: Exception) -> bool:
    """Is this an error that retrying will never fix?

    Quota exhaustion, invalid keys, and unknown model names all fail
    identically on every attempt. Rate limits and timeouts are worth a retry;
    these are not.
    """
    text = str(exc).lower()
    permanent_markers = (
        "insufficient_quota",
        "credit_balance_exhausted",
        "billing",
        "invalid_api_key",
        "invalid x-api-key",
        "authentication_error",
        "incorrect api key",
        "permission_denied",
        "model_not_found",
        "does not exist or you do not have access",
    )
    return any(marker in text for marker in permanent_markers)


def _explain(exc: Exception) -> str:
    """Translate a provider error into something the user can act on."""
    text = str(exc).lower()

    if "insufficient_quota" in text or "credit_balance_exhausted" in text or "billing" in text:
        return (
            "Your API account has no credit. An OpenAI or Anthropic API balance is "
            "separate from a ChatGPT or Claude subscription — add credit in the "
            "provider's billing settings. Statistics and charts still work without it."
        )

    if any(
        m in text
        for m in ("invalid_api_key", "invalid x-api-key", "authentication", "incorrect api key")
    ):
        return (
            "The API key was rejected. Check for a stray space or quote in your .env "
            "file, and that the key matches the LLM_PROVIDER you set."
        )

    if "model_not_found" in text or "does not exist" in text:
        return (
            "That model name isn't available on your account. Try a different "
            "LLM_MODEL in your .env file."
        )

    return str(exc)


def extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply.

    Handles the three things models actually do: clean JSON, JSON wrapped in
    ```json fences, and JSON buried in a sentence of explanation.
    """
    text = text.strip()

    # Strip markdown fences if present
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Last resort: grab the outermost braces
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"No parseable JSON in response: {exc}") from exc

    raise ValueError("No JSON object found in response")


def _stub_response(prompt: str) -> str:
    """Deterministic offline replies so the pipeline runs without an API key."""
    lowered = prompt.lower()

    if "recommend" in lowered or "plan" in lowered:
        return json.dumps(
            {
                "analyses": [
                    "descriptive_stats",
                    "correlation",
                    "distribution_tests",
                    "outlier_detection",
                ],
                "target_column": None,
                "concerns": ["Running offline — plan generated from heuristics, not an LLM."],
                "notes": "Set ANTHROPIC_API_KEY to enable model-driven planning.",
            }
        )

    return (
        "## Summary\n\n"
        "This report was generated in offline mode. The statistics, charts, and "
        "outlier detection below are real — only the written narrative is stubbed.\n\n"
        "Set `ANTHROPIC_API_KEY` in your `.env` file to get model-written analysis."
    )
