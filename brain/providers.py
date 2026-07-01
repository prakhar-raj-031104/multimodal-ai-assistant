"""
LLM provider abstraction — swap the whole brain with one env var.

Both providers expose the same `answer_stream(system, history, user_content)`
that yields response text, running agent tools first when needed:

  * GroqProvider     — Llama/GPT-OSS on Groq. Fast, cheap, weaker reasoning.
  * AnthropicProvider — Claude (Opus 4.8 by default). Strongest reasoning; needs
                        an ANTHROPIC_API_KEY (a Claude Pro subscription does NOT
                        grant API access — that's billed separately).

Set LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY to switch to Claude.
"""
from __future__ import annotations

import json
import os
from typing import Iterator, List, Optional

from core.config import LLMConfig
from core.logging_setup import get_logger
from brain.llm_engine import GroqLLM
from brain.reasoner import Reasoner

log = get_logger("brain.provider")


class GroqProvider:
    def __init__(self, cfg: LLMConfig, groq_client=None, tools=None) -> None:
        self.cfg = cfg
        self.reasoner = Reasoner(GroqLLM(cfg, client=groq_client), cfg, tools=tools)

    def answer_stream(self, system: str, history: List[dict],
                      user_content: str) -> Iterator[str]:
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_content})
        yield from self.reasoner.answer_stream(messages)


class AnthropicProvider:
    MAX_TOOL_ITERS = 3

    def __init__(self, cfg: LLMConfig, tools=None) -> None:
        self.cfg = cfg
        self.tools = tools
        self._client = None
        key = os.getenv(cfg.anthropic_api_key_env)
        if not key:
            log.error("%s not set — Claude provider unavailable", cfg.anthropic_api_key_env)
            return
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=key)
            log.info("Claude provider ready (%s)", cfg.anthropic_model)
        except Exception as e:  # noqa
            log.error("failed to init Anthropic client: %s", e)

    @property
    def available(self) -> bool:
        return self._client is not None

    def _anthropic_tools(self) -> Optional[list]:
        if not (self.cfg.enable_tools and self.tools and len(self.tools) > 0):
            return None
        out = []
        for s in self.tools.schemas():          # OpenAI-format -> Anthropic-format
            fn = s["function"]
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        return out

    def answer_stream(self, system: str, history: List[dict],
                      user_content: str) -> Iterator[str]:
        if self._client is None:
            yield "Claude is unavailable (missing ANTHROPIC_API_KEY)."
            return
        messages = list(history) + [{"role": "user", "content": user_content}]
        tools = self._anthropic_tools()

        # Resolve any tool calls first (non-streaming), then stream/emit the answer.
        if tools:
            for _ in range(self.MAX_TOOL_ITERS):
                resp = self._client.messages.create(
                    model=self.cfg.anthropic_model,
                    max_tokens=self.cfg.max_tokens,
                    system=system,
                    messages=messages,
                    tools=tools,
                )
                if resp.stop_reason != "tool_use":
                    for block in resp.content:
                        if getattr(block, "type", None) == "text":
                            yield block.text
                    return
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for block in resp.content:
                    if getattr(block, "type", None) == "tool_use":
                        out = self.tools.call(block.name, dict(block.input or {}))
                        log.info("🛠️  %s -> %s", block.name, str(out)[:60])
                        results.append({"type": "tool_result",
                                        "tool_use_id": block.id, "content": str(out)})
                messages.append({"role": "user", "content": results})

        # No tools (or tool loop exhausted): stream the final answer token-by-token.
        try:
            with self._client.messages.stream(
                model=self.cfg.anthropic_model,
                max_tokens=self.cfg.max_tokens,
                system=system,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:  # noqa
            log.error("Claude stream failed: %s", e)
            yield " (…response interrupted.)"


class GeminiProvider:
    """
    Google Gemini — the strongest *free* option. Get a no-cost, no-credit-card
    key at aistudio.google.com and set GEMINI_API_KEY. Free tier limits are far
    more generous than Groq's daily token cap, and Gemini 2.5 Flash is a strong,
    low-latency model. Tools aren't wired here (memory is still injected as
    context), so it's a clean conversational + vision-grounded brain.
    """

    def __init__(self, cfg: LLMConfig, tools=None) -> None:
        self.cfg = cfg
        self._client = None
        self._types = None
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            log.error("GEMINI_API_KEY not set — Gemini provider unavailable")
            return
        try:
            from google import genai
            from google.genai import types
            self._client = genai.Client(api_key=key)
            self._types = types
            log.info("Gemini provider ready (%s)", cfg.gemini_model)
        except Exception as e:  # noqa
            log.error("failed to init Gemini client: %s", e)

    @property
    def available(self) -> bool:
        return self._client is not None

    def answer_stream(self, system: str, history: List[dict],
                      user_content: str) -> Iterator[str]:
        if self._client is None:
            yield "Gemini is unavailable (missing GEMINI_API_KEY)."
            return
        types = self._types
        contents = []
        for m in history:
            role = "model" if m["role"] == "assistant" else "user"
            contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=user_content)]))
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self.cfg.gemini_model, contents=contents, config=config):
                if getattr(chunk, "text", None):
                    yield chunk.text
        except Exception as e:  # noqa
            log.error("Gemini stream failed: %s", e)
            yield " (…response interrupted.)"


def build_provider(cfg: LLMConfig, groq_client=None, tools=None):
    """Pick the provider from config, with a safe fallback to Groq."""
    if cfg.provider == "anthropic":
        prov = AnthropicProvider(cfg, tools=tools)
        if prov.available:
            return prov
        log.warning("Anthropic unavailable — falling back to Groq")
    elif cfg.provider == "gemini":
        prov = GeminiProvider(cfg, tools=tools)
        if prov.available:
            return prov
        log.warning("Gemini unavailable — falling back to Groq")
    return GroqProvider(cfg, groq_client=groq_client, tools=tools)
