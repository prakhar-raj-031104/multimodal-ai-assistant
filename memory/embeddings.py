"""
Text embedding provider with graceful fallback.

sentence-transformers (best) -> fastembed (fast ONNX) -> hashing embedding
(dependency-free, low quality but keeps the whole RAG path functional even on a
bare install). All backends expose the same `.encode(list[str]) -> np.ndarray`.
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from core.config import MemoryConfig
from core.logging_setup import get_logger

log = get_logger("memory.embeddings")


class Embeddings:
    def __init__(self, cfg: MemoryConfig) -> None:
        self.cfg = cfg
        self.dim = cfg.embedding_dim
        self._model = None
        self._kind = self._select(cfg.embedding_backend)

    def _select(self, preferred: str) -> str:
        order = {
            "sentence_transformers": [self._try_st, self._try_fastembed],
            "fastembed": [self._try_fastembed, self._try_st],
            "hash": [],
        }.get(preferred, [self._try_st, self._try_fastembed])
        for factory in order:
            kind = factory()
            if kind:
                log.info("embeddings: %s (dim=%d)", kind, self.dim)
                return kind
        log.warning("embeddings: hashing fallback (install sentence-transformers for quality)")
        return "hash"

    def _try_st(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.cfg.embedding_model)
            self.dim = self._model.get_sentence_embedding_dimension()
            return "sentence_transformers"
        except Exception as e:  # noqa
            log.debug("sentence-transformers unavailable: %s", e)
            return None

    def _try_fastembed(self):
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding()
            probe = next(iter(self._model.embed(["x"])))
            self.dim = len(probe)
            return "fastembed"
        except Exception as e:  # noqa
            log.debug("fastembed unavailable: %s", e)
            return None

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._kind == "sentence_transformers":
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        if self._kind == "fastembed":
            vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
            return _l2norm(vecs)
        return self._hash_encode(texts)

    def _hash_encode(self, texts: List[str]) -> np.ndarray:
        """Deterministic bag-of-hashed-tokens embedding. Not great, but real."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return _l2norm(out)


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)
