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
from typing import List


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

    def recent(self, limit: int = 12) -> List[Episode]:
        self._prune()
        return list(self._items)[-limit:]

    def as_context(self, limit: int = 12) -> str:
        eps = self.recent(limit)
        if not eps:
            return ""
        lines = []
        for e in eps:
            ago = int(e.age())
            tag = {"perception": "SAW", "utterance": "USER", "reply": "YOU",
                   "ambient": "HEARD"}.get(e.kind, e.kind.upper())
            lines.append(f"[{ago}s ago] {tag}: {e.text}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._items)
