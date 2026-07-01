"""
LLM engine (Groq) — streaming + tool-calling.

Wraps the Groq client with two primitives the rest of the brain builds on:
  * complete(...)      — single non-streaming call (used for routing/tools)
  * stream(...)        — yields text chunks as they arrive (low perceived latency)

A shared client is created once and reused everywhere (LLM, STT, TTS all ride
the same Groq key/connection pool).
"""
from __future__ import annotations

import os
import time
from typing import Iterator, List, Optional

from core.config import LLMConfig
from core.logging_setup import get_logger

log = get_logger("llm")

_CLIENT = None


def get_groq_client(api_key: Optional[str] = None):
    """Shared Groq client (also used by STT and TTS). None if no key."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        log.error("GROQ_API_KEY not set — LLM/STT/TTS cloud paths disabled")
        return None
    try:
        from groq import Groq
        _CLIENT = Groq(api_key=key)
    except Exception as e:  # noqa
        log.error("failed to init Groq client: %s", e)
        _CLIENT = None
    return _CLIENT


class GroqLLM:
    def __init__(self, cfg: LLMConfig, client=None) -> None:
        self.cfg = cfg
        self.client = client or get_groq_client()

    def complete(self, messages: List[dict], model: Optional[str] = None,
                 tools: Optional[list] = None, temperature: Optional[float] = None,
                 retries: int = 2):
        """Non-streaming completion. Returns the raw message object (may hold tool_calls)."""
        if self.client is None:
            return None
        kwargs = dict(
            model=model or self.cfg.model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens,
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        for attempt in range(retries + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message
            except Exception as e:  # noqa
                log.warning("LLM complete attempt %d failed: %s", attempt + 1, e)
                time.sleep(0.5 * (attempt + 1))
        return None

    def stream(self, messages: List[dict], model: Optional[str] = None,
               temperature: Optional[float] = None) -> Iterator[str]:
        """Yield response text chunks as they stream in."""
        if self.client is None:
            yield "Language model unavailable (missing GROQ_API_KEY)."
            return
        try:
            stream = self.client.chat.completions.create(
                model=model or self.cfg.model,
                messages=messages,
                temperature=self.cfg.temperature if temperature is None else temperature,
                max_tokens=self.cfg.max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:  # noqa
            log.error("LLM stream failed: %s", e)
            yield " (…response interrupted.)"
