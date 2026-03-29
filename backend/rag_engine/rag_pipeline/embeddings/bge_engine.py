from __future__ import annotations

import hashlib
import logging
from typing import List, Dict, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from rag_engine.rag_pipeline.chunking.semantic_chunker import TextChunk

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024
BATCH_SIZE = 32
MAX_SEQ_LEN = 512
BGE_QUERY_PREFIX = "Represent this sentence: "


# ─── Embedded chunk ──────────────────────────────────────────────────────────

class EmbeddedChunk:
    """A TextChunk paired with its embedding vector."""
    __slots__ = ("chunk", "vector")

    def __init__(self, chunk: TextChunk, vector: np.ndarray):
        self.chunk  = chunk
        self.vector = vector   


# ─── Embedding engine ────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Wraps BAAI/bge-small-en-v1.5 with:
      · Automatic GPU/CPU selection
      · Batched encoding
      · L2 normalization
      · Query vs document mode
      · In-memory dedup cache
    """

    def __init__(
        self,
        model_name:  str = MODEL_NAME,
        batch_size:  int = BATCH_SIZE,
        cache:       bool = True,
        device:      Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self._cache_enabled = cache
        self._cache: Dict[str, np.ndarray] = {}  

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        logger.info(f"Loading embedding model '{model_name}' on {device}…")
        self._model = SentenceTransformer(model_name, device=device)
        self._model.max_seq_length = MAX_SEQ_LEN
        logger.info("Embedding model ready.")

    # ── Public API ────────────────────────────────────────────────────────

    def embed_chunks(self, chunks: List[TextChunk]) -> List[EmbeddedChunk]:
        """
        Embed a list of TextChunks (document mode — no instruction prefix).
        Returns EmbeddedChunk list in the same order.
        """
        texts     = [c.text for c in chunks]
        vectors   = self._encode_texts(texts, mode="document")
        return [EmbeddedChunk(chunk, vec) for chunk, vec in zip(chunks, vectors)]

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query (query mode — BGE instruction prefix added).
        Returns L2-normalized float32 vector of shape (EMBEDDING_DIM,).
        """
        vectors = self._encode_texts([query], mode="query")
        return vectors[0]

    def embed_queries(self, queries: List[str]) -> np.ndarray:
        """Batch-embed multiple queries. Returns (N, EMBEDDING_DIM) array."""
        return self._encode_texts(queries, mode="query")

    # ── Internal ──────────────────────────────────────────────────────────

    def _encode_texts(self, texts: List[str], mode: str) -> np.ndarray:
        """
        Core encoding loop.
          mode='document' → plain text
          mode='query'    → BGE instruction prefix prepended
        Returns (N, EMBEDDING_DIM) float32 array.
        """
        if mode == "query":
            texts = [BGE_QUERY_PREFIX + t for t in texts]

        results    = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        to_encode  = []   

        
        for i, text in enumerate(texts):
            key = self._cache_key(text)
            if self._cache_enabled and key in self._cache:
                results[i] = self._cache[key]
            else:
                to_encode.append((i, text, key))

        
        if to_encode:
            idxs      = [t[0] for t in to_encode]
            raw_texts = [t[1] for t in to_encode]
            keys      = [t[2] for t in to_encode]

            for batch_start in range(0, len(raw_texts), self.batch_size):
                batch = raw_texts[batch_start: batch_start + self.batch_size]
                vecs  = self._model.encode(
                    batch,
                    batch_size          = self.batch_size,
                    show_progress_bar   = len(batch) > 100,
                    convert_to_numpy    = True,
                    normalize_embeddings= True,   
                )
                for j, vec in enumerate(vecs):
                    global_idx = idxs[batch_start + j]
                    results[global_idx] = vec
                    if self._cache_enabled:
                        self._cache[keys[batch_start + j]] = vec

        return results

    @staticmethod
    def _cache_key(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    # ── Diagnostics ───────────────────────────────────────────────────────

    def similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity (= dot product since vectors are L2-normalized)."""
        return float(np.dot(vec_a, vec_b))

    def cache_stats(self) -> dict:
        return {"cached_vectors": len(self._cache), "model": self.model_name}
