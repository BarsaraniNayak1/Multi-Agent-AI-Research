from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except ImportError:  # pragma: no cover - dependency is installed in project environments
    Groq = None


class GroqJSONClient:
    def __init__(self, retries: int = 3, backoff_seconds: float = 0.5) -> None:
        self.api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.client = Groq(api_key=self.api_key) if self.api_key and Groq is not None else None

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self.client is None:
            raise RuntimeError("GROQ_API_KEY is not configured")

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {"parse_error": "Model returned a non-object JSON value."}
            except Exception as error:
                last_error = error
                if attempt < self.retries - 1:
                    time.sleep(self.backoff_seconds * (2**attempt))
        return {"llm_error": str(last_error or "Unknown Groq error")}


def call_json_or_fallback(
    system_prompt: str,
    user_prompt: str,
    fallback: Callable[[], dict[str, Any]],
    client: GroqJSONClient | None = None,
) -> dict[str, Any]:
    active_client = client or GroqJSONClient()
    if not active_client.enabled:
        return fallback()
    result = active_client.complete_json(system_prompt, user_prompt)
    if "llm_error" in result or "parse_error" in result:
        fallback_result = fallback()
        fallback_result["llm_error"] = result.get("llm_error", result.get("parse_error"))
        return fallback_result
    return result