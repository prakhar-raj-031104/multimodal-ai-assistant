"""
Persistent vector store for long-term (semantic) memory.

Default backend is a dependency-free NumPy cosine index that persists to disk
as .npy + .jsonl — zero external services, works everywhere, fast enough for the
tens-of-thousands of memories a personal assistant accumulates. An optional
FAISS backend kicks in automatically for larger corpora when installed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

import numpy as np

from core.logging_setup import get_logger

log = get_logger("memory.vectorstore")


@dataclass
class MemoryRecord:
    id: str
    text: str
    kind: str = "fact"          # fact | perception | utterance | reminder
    ts: float = field(default_factory=time.time)
    meta: dict = field(default_factory=dict)


class VectorStore:
    def __init__(self, persist_dir: str, dim: int) -> None:
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = dim
        self._vecs = np.zeros((0, dim), dtype=np.float32)
        self._records: List[MemoryRecord] = []
        self._faiss = None
        self._vec_path = self.dir / "vectors.npy"
        self._meta_path = self.dir / "records.jsonl"
        self._load()
        self._maybe_faiss()

    # -- persistence --------------------------------------------------------
    def _load(self) -> None:
        try:
            if self._vec_path.exists() and self._meta_path.exists():
                self._vecs = np.load(self._vec_path)
                with open(self._meta_path) as f:
                    self._records = [MemoryRecord(**json.loads(l)) for l in f if l.strip()]
                if self._vecs.shape[0] != len(self._records):
                    log.warning("vector/record mismatch; resetting store")
                    self._reset()
                elif self._vecs.size and self._vecs.shape[1] != self.dim:
                    # The embedding backend changed under an existing store
                    # (e.g. sentence-transformers -> fastembed on deploy).
                    # Loading it anyway means every search raises a matmul
                    # shape error on the first question, so archive and restart.
                    old = self._vecs.shape[1]
                    self._archive()
                    log.warning("embedding dim changed %d -> %d; archived the old "
                                "store and started fresh (re-add memories to "
                                "re-embed them)", old, self.dim)
                    self._reset()
                else:
                    log.info("loaded %d memories", len(self._records))
        except Exception as e:  # noqa
            log.warning("could not load memory store (%s); starting fresh", e)
            self._vecs = np.zeros((0, self.dim), dtype=np.float32)
            self._records = []

    def _reset(self) -> None:
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)
        self._records = []

    def _archive(self) -> None:
        """Move an incompatible store aside rather than overwriting it — the
        text is still there, and re-embedding it is a choice the user can make."""
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in (self._vec_path, self._meta_path):
            if path.exists():
                try:
                    path.rename(path.with_suffix(path.suffix + f".{stamp}.bak"))
                except OSError as e:  # noqa
                    log.error("could not archive %s: %s", path.name, e)

    def _persist(self) -> None:
        try:
            np.save(self._vec_path, self._vecs)
            tmp = self._meta_path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                for r in self._records:
                    f.write(json.dumps(asdict(r)) + "\n")
            os.replace(tmp, self._meta_path)
        except Exception as e:  # noqa
            log.error("persist failed: %s", e)

    def _maybe_faiss(self) -> None:
        if len(self._records) < 5000:
            return
        try:
            import faiss
            index = faiss.IndexFlatIP(self.dim)
            if len(self._vecs):
                index.add(self._vecs)
            self._faiss = index
            log.info("FAISS index active (%d vectors)", len(self._records))
        except Exception:
            self._faiss = None

    # -- api ----------------------------------------------------------------
    def add(self, vector: np.ndarray, record: MemoryRecord) -> None:
        vector = vector.reshape(1, -1).astype(np.float32)
        self._vecs = np.vstack([self._vecs, vector]) if len(self._vecs) else vector
        self._records.append(record)
        if self._faiss is not None:
            self._faiss.add(vector)
        self._persist()

    def touch(self, record_id: str) -> bool:
        """Refresh a record's timestamp — used when a duplicate fact is
        re-observed, so recency reflects the last time we actually saw it."""
        for r in self._records:
            if r.id == record_id:
                r.ts = time.time()
                self._persist()
                return True
        return False

    def delete(self, record_id: str) -> bool:
        """Remove a memory permanently (bad/expired facts must be evictable)."""
        for i, r in enumerate(self._records):
            if r.id == record_id:
                self._records.pop(i)
                self._vecs = np.delete(self._vecs, i, axis=0)
                self._faiss = None  # index no longer matches; rebuild lazily
                self._persist()
                self._maybe_faiss()
                return True
        return False

    def all(self) -> List[MemoryRecord]:
        return list(self._records)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> List[tuple]:
        if len(self._records) == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32)
        if self._faiss is not None:
            scores, idx = self._faiss.search(q, min(top_k, len(self._records)))
            return [(self._records[i], float(scores[0][j]))
                    for j, i in enumerate(idx[0]) if i >= 0]
        sims = (self._vecs @ q.T).ravel()
        top = np.argsort(-sims)[:top_k]
        return [(self._records[i], float(sims[i])) for i in top]

    def __len__(self) -> int:
        return len(self._records)
