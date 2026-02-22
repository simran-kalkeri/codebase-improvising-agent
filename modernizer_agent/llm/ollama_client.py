"""
Ollama LLM client for the Modernization Agent.

Sends prompts to a locally running Ollama instance and enforces
strict JSON-only responses. Non-JSON outputs are retried up to
LLM_JSON_RETRIES times before raising an error.

Usage:
    from modernizer_agent.llm.ollama_client import OllamaClient
    client = OllamaClient()
    response = client.generate(prompt="...")  # returns parsed dict
"""

import json
import re
from typing import Any

import requests

from modernizer_agent.config import (
    LLM_JSON_RETRIES,
    LLM_MAX_TOKENS,
    LLM_REQUEST_TIMEOUT,
    OLLAMA_GENERATE_ENDPOINT,
    OLLAMA_MODEL,
)
from modernizer_agent.utils.logger import get_logger

log = get_logger("modernizer_agent.llm")


class OllamaClientError(Exception):
    """Raised when the Ollama client encounters an unrecoverable error."""


class OllamaClient:
    """Thin wrapper around the Ollama HTTP API that guarantees JSON responses."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        endpoint: str = OLLAMA_GENERATE_ENDPOINT,
        max_tokens: int = LLM_MAX_TOKENS,
        timeout: int = LLM_REQUEST_TIMEOUT,
        json_retries: int = LLM_JSON_RETRIES,
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.json_retries = json_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: str = "") -> dict[str, Any]:
        """Send *prompt* to Ollama and return a **parsed JSON dict**.

        The model is instructed to reply with JSON only.  If the raw
        response is not valid JSON, the client retries up to
        ``self.json_retries`` times with a corrective nudge.

        Raises
        ------
        OllamaClientError
            If the model fails to produce valid JSON after all retries,
            or if the HTTP request itself fails.
        """
        last_error: str = ""

        for attempt in range(1, self.json_retries + 1):
            # On retries, append a corrective instruction.
            effective_prompt = prompt
            if attempt > 1:
                effective_prompt = (
                    f"{prompt}\n\n"
                    f"[SYSTEM: Your previous response was not valid JSON. "
                    f"Error: {last_error}. "
                    f"You MUST respond with valid JSON only. No markdown, "
                    f"no explanation text outside the JSON object.]"
                )

            raw_text = self._call_ollama(effective_prompt, system)

            # --- Attempt to parse JSON from the response ---------------
            parsed = self._extract_json(raw_text)
            if parsed is not None:
                log.info(
                    "LLM returned valid JSON",
                    extra={"tool": "ollama", "action": "generate", "result": "ok"},
                )
                return parsed

            last_error = f"Could not parse JSON from: {raw_text[:200]!r}"
            log.warning(
                "LLM returned non-JSON, retrying",
                extra={
                    "tool": "ollama",
                    "action": "generate",
                    "result": "retry",
                    "error": last_error,
                    "iteration": attempt,
                },
            )

        raise OllamaClientError(
            f"LLM failed to produce valid JSON after {self.json_retries} attempts. "
            f"Last error: {last_error}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_ollama(self, prompt: str, system: str) -> str:
        """Make the raw HTTP POST to Ollama and return the text response."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": self.max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise OllamaClientError(
                f"Cannot connect to Ollama at {self.endpoint}. "
                f"Is Ollama running? Error: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise OllamaClientError(
                f"Ollama request timed out after {self.timeout}s: {exc}"
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaClientError(
                f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}"
            ) from exc

        body = resp.json()
        return body.get("response", "")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        """Try to extract a JSON object from *text*.

        Handles three common patterns:
        1. The entire text is valid JSON.
        2. JSON is wrapped in a ```json ... ``` code fence.
        3. A JSON object is embedded somewhere in the text.

        Returns ``None`` if no valid JSON object can be found.
        """
        # --- Strategy 1: direct parse ---
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

        # --- Strategy 2: strip markdown code fences ---
        fence_pattern = r"```(?:json)?\s*\n?(.*?)\n?\s*```"
        fence_match = re.search(fence_pattern, text, re.DOTALL)
        if fence_match:
            try:
                obj = json.loads(fence_match.group(1).strip())
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

        # --- Strategy 3: find first { ... } block ---
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            try:
                obj = json.loads(text[brace_start : brace_end + 1])
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass

        return None
