"""
Query router — decides what to do with a transcribed utterance.

Not everything the mic hears deserves a full LLM turn. Cheap heuristics catch
the obvious cases instantly (0ms); only genuinely ambiguous utterances fall
through to the fast model. Classes:
  * QUESTION / COMMAND — answer it (full pipeline)
  * VISUAL             — needs a fresh look at the scene before answering
  * AMBIENT            — overheard chatter; log to memory, don't respond
"""
from __future__ import annotations

import re
from enum import Enum

from core.config import LLMConfig
from core.logging_setup import get_logger

log = get_logger("brain.router")


class Intent(str, Enum):
    ANSWER = "answer"
    VISUAL = "visual"
    AMBIENT = "ambient"


_VISUAL_HINTS = re.compile(
    r"\b(see|look|looking|show|read|what('| i)s this|in front|around me|"
    r"this (object|thing|sign|text|label)|point(ing)? at|holding|color|colour)\b",
    re.I,
)
_DIRECT_HINTS = re.compile(
    r"\b(you|hey|assistant|jarvis|help|tell me|what|who|when|where|why|how|"
    r"can you|could you|remind|remember|search|find|explain|translate|"
    r"summar|note)\b",
    re.I,
)


class Router:
    def __init__(self, llm=None, cfg: LLMConfig = None) -> None:
        self.llm = llm
        self.cfg = cfg

    def route(self, text: str) -> Intent:
        t = text.strip()
        if len(t) < 2:
            return Intent.AMBIENT
        if _VISUAL_HINTS.search(t):
            return Intent.VISUAL
        # A question mark or an imperative/direct address -> answer it.
        if t.endswith("?") or _DIRECT_HINTS.search(t):
            return Intent.ANSWER
        # Very short, non-directed speech is probably ambient chatter.
        if len(t.split()) <= 3:
            return Intent.AMBIENT
        # Ambiguous longer utterance: default to answering (better UX than
        # silently ignoring). A model-based classifier could refine this.
        return Intent.ANSWER
