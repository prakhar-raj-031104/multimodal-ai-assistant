"""
Event types and a minimal async pub/sub bus.

The whole system is a set of independent async producers (audio, vision) and
consumers (memory, brain) that never call each other directly — they publish
and subscribe to typed events on this bus. That decoupling is what makes the
runtime "always-on" and easy to extend.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    PERCEPTION = "perception"      # a new visual understanding of the scene
    SPEECH_START = "speech_start"  # user started speaking (used for barge-in)
    UTTERANCE = "utterance"        # a finished, transcribed user utterance
    ASSISTANT_REPLY = "reply"      # assistant produced a final response
    AMBIENT_SOUND = "ambient"      # non-speech audio event
    SYSTEM = "system"              # lifecycle / health


@dataclass
class Event:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class Perception:
    """One structured visual snapshot of the surroundings."""
    summary: str
    raw: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def as_text(self) -> str:
        return self.summary


@dataclass
class Utterance:
    text: str
    is_final: bool = True
    speaker: Optional[str] = None
    ts: float = field(default_factory=time.time)


class EventBus:
    """Fan-out async bus. Each subscriber gets its own queue."""

    def __init__(self) -> None:
        self._subs: List[asyncio.Queue] = []

    def subscribe(self, maxsize: int = 100) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        self._subs.append(q)
        return q

    async def publish(self, event: Event) -> None:
        for q in self._subs:
            # Drop-oldest policy: real-time systems must never block producers.
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            await q.put(event)
