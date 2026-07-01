"""
The agentic reasoning loop.

Given fused context + conversation history, produce a spoken response. Flow
optimised for latency:

  1. One tool-aware call on the fast model. If the model wants tools, run them
     and loop (bounded). This resolves "what time is it / recall / search".
  2. Once no more tools are needed, STREAM the final answer on the main model so
     TTS can start speaking the first sentence almost immediately.

If tools are disabled, we skip straight to streaming — the lowest-latency path.
"""
from __future__ import annotations

import json
from typing import Iterator, List, Optional

from core.config import LLMConfig
from core.logging_setup import get_logger
from brain.llm_engine import GroqLLM

log = get_logger("brain.reasoner")

MAX_TOOL_ITERS = 3


class Reasoner:
    def __init__(self, llm: GroqLLM, cfg: LLMConfig, tools=None) -> None:
        self.llm = llm
        self.cfg = cfg
        self.tools = tools

    def answer_stream(self, messages: List[dict]) -> Iterator[str]:
        """Yield final-answer text chunks, running tools first if needed."""
        working = list(messages)

        if self.cfg.enable_tools and self.tools and len(self.tools) > 0:
            working = self._run_tool_loop(working)

        yield from self.llm.stream(working, model=self.cfg.model)

    def _run_tool_loop(self, messages: List[dict]) -> List[dict]:
        schemas = self.tools.schemas()
        for _ in range(MAX_TOOL_ITERS):
            msg = self.llm.complete(messages, model=self.cfg.model, tools=schemas)
            if msg is None:
                break
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                # No tools needed. Fold any content back so streaming can restart
                # cleanly from the same context (idempotent).
                break
            # Record the assistant's tool request, then each tool result.
            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in tool_calls
                ],
            })
            for tc in tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except Exception:
                    args = {}
                result = self.tools.call(tc.function.name, args)
                log.info("🛠️  %s(%s) -> %s", tc.function.name, args, str(result)[:60])
                messages.append({
                    "role": "tool", "tool_call_id": tc.id,
                    "name": tc.function.name, "content": str(result),
                })
        return messages
