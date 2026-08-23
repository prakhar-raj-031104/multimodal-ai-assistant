"""
The agentic reasoning loop.

Given fused context + conversation history, produce a spoken response.

Latency design — one streaming pass per model turn:

  1. Stream the answer with the tool schemas attached. If the model just wants
     to talk (the common case), its tokens go straight to TTS: **one** LLM call,
     first audio in a few hundred ms.
  2. If it asks for tools instead, the stream ends in tool calls; we run them,
     append the results, and stream again.

The previous version always made a blocking, non-streaming "does it need tools?"
call, threw that answer away, and then regenerated the identical reply with a
second streaming call — every single turn paid for two full generations, and the
wasted call grew with conversation length (measured 6-8s on turn 8).
"""
from __future__ import annotations

import json
from typing import Iterator, List, Optional

from core.config import LLMConfig
from core.logging_setup import get_logger
from brain.llm_engine import GroqLLM

log = get_logger("brain.reasoner")

MAX_TOOL_ITERS = 3

# If the model has already streamed this much of a real answer, that answer has
# been spoken aloud — any further tool call it tacks on is redundant, and
# looping would make it say the whole thing a second time (observed: the model
# re-calling get_current_time after already giving the time, producing
# "It's 2:51 a.m....It's 2:51 a.m....").
_ANSWER_COMMITTED_CHARS = 40


class Reasoner:
    def __init__(self, llm: GroqLLM, cfg: LLMConfig, tools=None) -> None:
        self.llm = llm
        self.cfg = cfg
        self.tools = tools

    def _schemas(self) -> Optional[list]:
        if self.cfg.enable_tools and self.tools and len(self.tools) > 0:
            return self.tools.schemas()
        return None

    def answer_stream(self, messages: List[dict]) -> Iterator[str]:
        """Yield final-answer text chunks, running tools in between if needed."""
        working = list(messages)
        schemas = self._schemas()

        if schemas is None:
            yield from self.llm.stream(working, model=self.cfg.model)
            return

        executed: set = set()   # (name, args) already run this turn
        tool_rounds = 0

        for _ in range(MAX_TOOL_ITERS):
            calls: List[dict] = []
            spoken = 0
            for ev in self.llm.stream_events(working, tools=schemas, model=self.cfg.model):
                if ev["type"] == "text":
                    spoken += len(ev["text"])
                    yield ev["text"]
                elif ev["type"] == "tool_calls":
                    calls = ev["calls"]

            if not calls:
                return  # answered directly — one LLM call for this turn

            # The model already delivered an answer and is now tacking on more
            # tool calls. Looping here makes it keep talking: observed
            # "Got it, I'll remember that." -> "Sure thing!" -> "You're welcome!
            # Let me know if there's anything else", one line per iteration, all
            # of it spoken aloud. Once it has answered, we are done.
            if spoken and (tool_rounds or spoken >= _ANSWER_COMMITTED_CHARS):
                log.debug("answer already streamed (%d chars); dropping %d "
                          "trailing tool call(s)", spoken, len(calls))
                return

            fresh = [c for c in calls
                     if (c["name"], (c["args"] or "").strip()) not in executed]
            if not fresh:
                # Every call is a verbatim repeat of one we just ran — the model
                # is stuck. Force a plain answer instead of burning iterations.
                log.debug("all %d tool call(s) already executed; forcing answer",
                          len(calls))
                break

            working.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["args"] or "{}"}}
                    for c in fresh
                ],
            })
            for c in fresh:
                executed.add((c["name"], (c["args"] or "").strip()))
                try:
                    args = json.loads(c["args"] or "{}")
                except Exception:
                    args = {}
                result = self.tools.call(c["name"], args)
                log.info("🛠️  %s(%s) -> %s", c["name"], args, str(result)[:60])
                working.append({
                    "role": "tool", "tool_call_id": c["id"],
                    "name": c["name"], "content": str(result),
                })
            tool_rounds += 1
            if spoken:
                # Short preamble ("let me check…") already went to the speaker.
                working.append({
                    "role": "system",
                    "content": "You already spoke a preamble aloud. Give only the "
                               "answer now; do not repeat what you already said.",
                })

        # Tool budget exhausted, or the model looped — answer from what we have.
        yield from self.llm.stream(working, model=self.cfg.model)
