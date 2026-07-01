"""
Memory manager: the assistant's hippocampus.

Ties together the three tiers:
  * episodic buffer  — last ~2 min of raw events (instant, no retrieval)
  * vector store     — durable semantic facts (RAG retrieval)
  * consolidation    — periodically distils episodic -> durable facts

The brain calls `retrieve(query)` to fuse recent context + relevant long-term
memory, and `observe(...)` to feed perceptions/utterances in. Consolidation runs
opportunistically off the hot path.
"""
from __future__ import annotations

import time
import uuid
from typing import List, Optional

from core.config import MemoryConfig, LLMConfig
from core.logging_setup import get_logger
from memory.embeddings import Embeddings
from memory.episodic import EpisodicBuffer
from memory.vector_store import VectorStore, MemoryRecord
from memory.consolidation import consolidate

log = get_logger("memory.manager")


class MemoryManager:
    def __init__(self, mcfg: MemoryConfig, llm_cfg: LLMConfig, groq_client=None) -> None:
        self.mcfg = mcfg
        self.llm_cfg = llm_cfg
        self._groq = groq_client
        self.embeddings = Embeddings(mcfg)
        self.store = VectorStore(mcfg.persist_dir, self.embeddings.dim)
        self.episodic = EpisodicBuffer(mcfg.episodic_window_s)
        self._since_consolidation = 0

    # -- ingest -------------------------------------------------------------
    def observe(self, kind: str, text: str) -> None:
        if not text or not text.strip():
            return
        self.episodic.add(kind, text)
        if kind in ("perception", "utterance"):
            self._since_consolidation += 1

    def remember_fact(self, text: str, kind: str = "fact", meta: Optional[dict] = None) -> None:
        """Directly commit a durable memory (used by consolidation & tools)."""
        vec = self.embeddings.encode([text])[0]
        rec = MemoryRecord(id=str(uuid.uuid4()), text=text, kind=kind, meta=meta or {})
        self.store.add(vec, rec)

    # -- retrieve -----------------------------------------------------------
    def retrieve(self, query: str, top_k: Optional[int] = None) -> dict:
        top_k = top_k or self.mcfg.top_k
        recent = self.episodic.as_context()
        facts: List[str] = []
        if query and len(self.store) > 0:
            qv = self.embeddings.encode([query])[0]
            hits = self.store.search(qv, top_k)
            # Higher threshold = only genuinely relevant memories reach the LLM.
            # Loosely-related facts injected as "MEMORY" are a hallucination source.
            facts = [f"{r.text}" for r, score in hits if score > 0.30]
        return {"recent": recent, "facts": facts}

    def format_context(self, query: str) -> str:
        r = self.retrieve(query)
        parts = []
        if r["recent"]:
            parts.append("RECENT (last couple minutes):\n" + r["recent"])
        if r["facts"]:
            parts.append("RELEVANT MEMORY:\n" + "\n".join(f"- {f}" for f in r["facts"]))
        return "\n\n".join(parts)

    # -- consolidation (off hot path) --------------------------------------
    def maybe_consolidate(self) -> int:
        if self._since_consolidation < self.mcfg.consolidate_every:
            return 0
        self._since_consolidation = 0
        log_text = self.episodic.as_context(limit=self.mcfg.consolidate_every)
        facts = consolidate(log_text, self._groq, self.llm_cfg)
        for f in facts:
            self.remember_fact(f, kind="fact", meta={"source": "consolidation", "ts": time.time()})
        if facts:
            log.info("🧠 consolidated %d durable fact(s)", len(facts))
        return len(facts)
