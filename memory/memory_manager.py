"""
Memory manager: the assistant's hippocampus.

Ties together the three tiers:
  * episodic buffer  — last ~2 min of raw events (instant, no retrieval)
  * vector store     — durable semantic facts (RAG retrieval)
  * consolidation    — periodically distils episodic -> durable facts

Retrieval policy (this is what keeps the assistant from hallucinating):

  1. **Retrieve only when the question actually needs memory.** "What is the
     capital of France?" must not drag in "The user's favourite colour is teal".
     A cheap lexical gate decides; the LLM can always force a lookup itself via
     the `recall_memory` tool.
  2. **High relevance bar.** Only facts that clearly beat
     `relevance_threshold` reach the prompt. Loosely-related facts injected as
     "MEMORY" are the single biggest source of confident nonsense.
  3. **Never present memory as the present.** Facts are rendered with their age
     and flagged stale past `stale_after_s`, so the model can't report a
     remembered thing as something happening now.
  4. **The conversation is not memory.** Dialogue is passed to the model as
     native chat turns, so it is excluded from the context block rather than
     duplicated into it.
  5. **Write-side hygiene.** Near-duplicate facts are dropped instead of
     appended, so the store doesn't drift into 50 restatements of one thing.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import List, Optional

import numpy as np

from core.config import MemoryConfig, LLMConfig
from core.logging_setup import get_logger
from memory.embeddings import Embeddings
from memory.episodic import EpisodicBuffer, DIALOGUE_KINDS
from memory.vector_store import VectorStore, MemoryRecord
from memory.consolidation import consolidate

log = get_logger("memory.manager")

# Questions that plausibly depend on stored personal history. Anything else
# skips long-term retrieval entirely (cheaper AND less hallucination surface).
_RECALL_HINTS = re.compile(
    r"\b(remember|remembered|recall|forget|forgot|memor(y|ies)|"
    r"earlier|before|previously|yesterday|last (time|week|night|month)|ago|"
    r"my|mine|our|i (said|told|asked|mentioned|had|was|am|like|prefer|need)|"
    r"did i|do i|have i|was i|am i|who is|who was|what did|what was|where did|"
    r"where was|when did|when was|remind|reminder|appointment|meeting|demo|"
    r"deadline|schedule|birthday|name of|favou?rite|usual|habit)\b",
    re.I,
)


def _fmt_age(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 5400:
        return f"{int(seconds // 60)}m ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)}h ago"
    return f"{int(seconds // 86400)}d ago"


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

    def remember_fact(self, text: str, kind: str = "fact",
                      meta: Optional[dict] = None) -> bool:
        """Commit a durable memory. Returns False if it was a near-duplicate.

        Deduping on write is what keeps long-term memory usable: without it,
        consolidation re-saves a paraphrase of the same fact every few minutes
        and retrieval starts returning five copies of one thing.
        """
        text = (text or "").strip()
        if not text:
            return False
        vec = self.embeddings.encode([text])[0]
        dup = self._nearest(vec)
        if dup is not None:
            rec, score = dup
            if score >= self.mcfg.dedupe_threshold:
                # Refresh the existing record's timestamp instead of appending.
                self.store.touch(rec.id)
                log.debug("dedupe: %r ~= %r (%.2f)", text[:40], rec.text[:40], score)
                return False
        rec = MemoryRecord(id=str(uuid.uuid4()), text=text, kind=kind, meta=meta or {})
        self.store.add(vec, rec)
        return True

    def _nearest(self, vec: np.ndarray):
        if len(self.store) == 0:
            return None
        hits = self.store.search(vec, 1)
        return hits[0] if hits else None

    # -- retrieve -----------------------------------------------------------
    def needs_recall(self, query: str) -> bool:
        """Should this query hit long-term memory at all?"""
        if self.mcfg.always_retrieve:
            return True
        q = (query or "").strip()
        if len(q) < 3:
            return False
        return bool(_RECALL_HINTS.search(q))

    def retrieve(self, query: str, top_k: Optional[int] = None,
                 force: bool = False) -> dict:
        """Fuse recent episodic context with (optionally) long-term facts.

        `force=True` bypasses the relevance gate — used by the `recall_memory`
        tool, where the model has explicitly decided it needs to look something
        up.
        """
        top_k = top_k or self.mcfg.top_k
        kinds = None
        if self.mcfg.exclude_dialogue_from_context:
            kinds = [k for k in ("perception", "utterance", "reply", "ambient")
                     if k not in DIALOGUE_KINDS]
        recent = self.episodic.as_context(kinds=kinds)

        facts: List[dict] = []
        if query and len(self.store) > 0 and (force or self.needs_recall(query)):
            qv = self.embeddings.encode([query])[0]
            hits = self.store.search(qv, top_k)
            threshold = self.mcfg.relevance_threshold
            for r, score in hits:
                if score < threshold:
                    continue
                facts.append({
                    "text": r.text,
                    "score": float(score),
                    "age_s": max(0.0, time.time() - r.ts),
                    "kind": r.kind,
                })
        return {"recent": recent, "facts": facts}

    def recent_facts(self, limit: int = 5) -> List[dict]:
        """Newest durable facts, ignoring semantic similarity.

        Meta-questions ("what did I ask you to remember?", "what do you know
        about me?") share almost no vocabulary with the facts they are asking
        about, so a pure similarity search returns nothing. Recency is the right
        index for those, and it is only ever reached through an explicit
        `recall_memory` call — never injected into a prompt unasked.
        """
        rows = sorted(self.store.all(), key=lambda r: r.ts, reverse=True)[:limit]
        return [{"text": r.text, "score": 0.0,
                 "age_s": max(0.0, time.time() - r.ts), "kind": r.kind}
                for r in rows]

    def format_context(self, query: str) -> str:
        """Render the memory block for the prompt. Empty string when there is
        nothing genuinely relevant — an empty block beats a padded one."""
        r = self.retrieve(query)
        parts = []
        if r["recent"]:
            parts.append(
                "WHAT YOU OBSERVED RECENTLY (past events, not the present):\n"
                + r["recent"])
        if r["facts"]:
            lines = []
            for f in r["facts"]:
                age = _fmt_age(f["age_s"])
                stale = " — may be outdated" if f["age_s"] > self.mcfg.stale_after_s else ""
                lines.append(f"- ({age}{stale}) {f['text']}")
            parts.append(
                "BACKGROUND FACTS FROM LONG-TERM MEMORY (recorded earlier; use "
                "only if they answer the question):\n" + "\n".join(lines))
        return "\n\n".join(parts)

    # -- consolidation (off hot path) --------------------------------------
    def maybe_consolidate(self) -> int:
        if self._since_consolidation < self.mcfg.consolidate_every:
            return 0
        self._since_consolidation = 0
        log_text = self.episodic.as_context(limit=self.mcfg.consolidate_every)
        facts = consolidate(log_text, self._groq, self.llm_cfg)
        stored = 0
        for f in facts:
            if self.remember_fact(f, kind="fact",
                                  meta={"source": "consolidation", "ts": time.time()}):
                stored += 1
        if stored:
            log.info("🧠 consolidated %d durable fact(s) (%d dropped as duplicates)",
                     stored, len(facts) - stored)
        return stored
