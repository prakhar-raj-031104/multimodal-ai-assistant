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
from typing import Dict, Iterator, List, Optional

from core.config import LLMConfig
from core.logging_setup import get_logger

log = get_logger("llm")

_CLIENT = None

# Models that accept the `reasoning_effort` parameter on Groq.
_REASONING_MODELS = ("gpt-oss", "qwen3")


def _supports_reasoning_effort(model: str) -> bool:
    m = (model or "").lower()
    return any(k in m for k in _REASONING_MODELS)


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

    def _effort(self, model: str) -> dict:
        eff = (self.cfg.reasoning_effort or "").strip().lower()
        if not eff or eff == "none" or not _supports_reasoning_effort(model):
            return {}
        return {"reasoning_effort": eff}

    def complete(self, messages: List[dict], model: Optional[str] = None,
                 tools: Optional[list] = None, temperature: Optional[float] = None,
                 retries: int = 2):
        """Non-streaming completion. Returns the raw message object (may hold tool_calls)."""
        if self.client is None:
            return None
        model = model or self.cfg.model
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens,
            **self._effort(model),
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
            model = model or self.cfg.model
            stream = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=self.cfg.temperature if temperature is None else temperature,
                max_tokens=self.cfg.max_tokens,
                stream=True,
                **self._effort(model),
            )
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:  # noqa
            log.error("LLM stream failed: %s", e)
            yield " (…response interrupted.)"

    def stream_events(self, messages: List[dict], tools: Optional[list] = None,
                      model: Optional[str] = None,
                      temperature: Optional[float] = None) -> Iterator[Dict]:
        """Single streaming pass that can end in EITHER text or tool calls.

        Yields {"type": "text", "text": ...} as tokens arrive and, if the model
        decided to call tools instead, a final {"type": "tool_calls", "calls":
        [...]}. This is what lets a no-tool turn cost ONE LLM call: the previous
        design always ran a blocking tool-probe call and then regenerated the
        same answer by streaming, paying for every reply twice.
        """
        if self.client is None:
            yield {"type": "text", "text": "Language model unavailable (missing GROQ_API_KEY)."}
            return
        model = model or self.cfg.model
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens,
            stream=True,
            **self._effort(model),
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        # Tool-call deltas arrive fragmented (name in one chunk, arguments split
        # across many), keyed by index — reassemble them before dispatching.
        acc: Dict[int, dict] = {}
        try:
            for chunk in self.client.chat.completions.create(**kwargs):
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta is None:
                    continue
                if getattr(delta, "content", None):
                    yield {"type": "text", "text": delta.content}
                for tc in (getattr(delta, "tool_calls", None) or []):
                    slot = acc.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                    if getattr(tc, "id", None):
                        slot["id"] = tc.id
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            slot["name"] = fn.name
                        if getattr(fn, "arguments", None):
                            slot["args"] += fn.arguments
        except Exception as e:  # noqa
            log.error("LLM stream_events failed: %s", e)
            if not acc:
                yield {"type": "text", "text": " (…response interrupted.)"}
            return

        calls = [acc[i] for i in sorted(acc) if acc[i]["name"]]
        if calls:
            yield {"type": "tool_calls", "calls": calls}
