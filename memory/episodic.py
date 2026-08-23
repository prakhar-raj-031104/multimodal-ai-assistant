"""
Episodic (short-term) memory: a rolling, time-bounded buffer of recent events.

This is the assistant's "what just happened in the last couple of minutes" —
recent perceptions and utterances, always available to the brain without a
retrieval call. Old items age out of the window; before they do, the memory
manager consolidates them into durable facts in the vector store.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

# Dialogue turns are already delivered to the model as native chat messages.
# Repeating them inside the context block makes models blend old turns into the
# current answer, so callers can filter them out with `kinds=`.
DIALOGUE_KINDS = ("utterance", "reply")


@dataclass
class Episode:
    kind: str            # perception | utterance | reply | ambient
    text: str
    ts: float = field(default_factory=time.time)

    def age(self) -> float:
        return time.time() - self.ts


class EpisodicBuffer:
    def __init__(self, window_s: float = 120.0, maxlen: int = 500) -> None:
        self.window_s = window_s
        self._items: deque[Episode] = deque(maxlen=maxlen)

    def add(self, kind: str, text: str) -> Episode:
        ep = Episode(kind=kind, text=text.strip())
        self._items.append(ep)
        return ep

    def _prune(self) -> None:
        now = time.time()
        while self._items and (now - self._items[0].ts) > self.window_s:
            self._items.popleft()

    def recent(self, limit: int = 12, kinds: Optional[Iterable[str]] = None) -> List[Episode]:
        self._prune()
        items = list(self._items)
        if kinds is not None:
            allow = set(kinds)
            items = [e for e in items if e.kind in allow]
        return items[-limit:]

    def as_context(self, limit: int = 12, kinds: Optional[Iterable[str]] = None,
                   max_chars: int = 220) -> str:
        """Render recent episodes as a compact, explicitly time-stamped block.

        `max_chars` truncates individual entries — a raw VLM JSON dump or a long
        monologue pasted verbatim drowns out the actual question.
        """
        eps = self.recent(limit, kinds=kinds)
        if not eps:
            return ""
        lines = []
        for e in eps:
            ago = int(e.age())
            tag = {"perception": "SAW", "utterance": "USER", "reply": "YOU",
                   "ambient": "HEARD"}.get(e.kind, e.kind.upper())
            text = e.text if len(e.text) <= max_chars else e.text[:max_chars].rstrip() + "…"
            lines.append(f"[{ago}s ago] {tag}: {text}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._items)
