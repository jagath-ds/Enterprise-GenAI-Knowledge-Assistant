"""
Retrieval Engine — Sprint 3
Turns a raw similarity search into a well-structured, LLM-ready context.

Pipeline:
  1. Embed query (with BGE instruction prefix)
  2. Similarity search → top-K candidates
  3. Score threshold filter  (discard irrelevant chunks)
  4. MMR (Maximal Marginal Relevance) re-rank (diversity)
  5. Build ordered context string with source citations
  6. Return RetrievedContext ready for prompt assembly
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from rag_engine.rag_pipeline.embeddings.bge_engine import EmbeddingEngine
from rag_engine.rag_pipeline.vectorstore.store import VectorStore, RetrievalResult

logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────

DEFAULT_TOP_K          = 3      # chunks to return to the LLM
DEFAULT_FETCH_K        = 25     # candidates fetched before re-ranking
MIN_RELEVANCE_SCORE    = 0.30   # below this → chunk is noise
MMR_LAMBDA             = 0.6    # 1.0 = pure relevance; 0.0 = pure diversity


# ─── Result container ────────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """Everything the LLM needs: context text + citation metadata."""
    query:          str
    chunks:         List[RetrievalResult]   # ordered, de-duplicated chunks
    context_text:   str                     # formatted string for prompt
    sources:        List[dict]              # citation list for response
    total_tokens:   int = 0
    strategy_used:  str = "similarity+mmr"

    def is_empty(self) -> bool:
        return len(self.chunks) == 0


# ─── Retrieval engine ────────────────────────────────────────────────────────

class RetrievalEngine:
    """
    Full retrieval pipeline: embed → search → filter → MMR → format.

    Usage:
        engine  = RetrievalEngine(embedding_engine, vector_store)
        context = engine.retrieve("What is the refund policy?")
        # pass context.context_text to LLM
    """

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        vector_store:     VectorStore,
        top_k:            int   = DEFAULT_TOP_K,
        fetch_k:          int   = DEFAULT_FETCH_K,
        min_score:        float = MIN_RELEVANCE_SCORE,
        mmr_lambda:       float = MMR_LAMBDA,
    ):
        self.embedder   = embedding_engine
        self.store      = vector_store
        self.top_k      = top_k
        self.fetch_k    = fetch_k
        self.min_score  = min_score
        self.mmr_lambda = mmr_lambda

    # ── Main entry point ──────────────────────────────────────────────────

    def retrieve(
        self,
        query:         str,
        top_k:         Optional[int]   = None,
        filter_doc_id: Optional[str]   = None,
    ) -> RetrievedContext:
        """
        Full pipeline: embed → search → filter → MMR → format context.
        Returns RetrievedContext (context_text is empty string if no good matches).
        """
        k = top_k or self.top_k

        # 1. Embed query
        query_vector = self.embedder.embed_query(query)

        # 2. Fetch more candidates than needed (pre-reranking pool)
        candidates = self.store.search(
            query_vector   = query_vector,
            top_k          = self.fetch_k,
            filter_doc_id  = filter_doc_id,
        )
        logger.debug(f"Fetched {len(candidates)} candidates for query: {query!r}")

        # 3. Score threshold filter
        candidates = [r for r in candidates if r.score >= self.min_score]
        if not candidates:
            logger.info("No chunks passed the relevance threshold.")
            return self._empty_context(query)

        # 4. MMR re-rank for diversity
        selected = self._mmr_rerank(candidates, query_vector, k)

        # 5. Sort selected chunks by document + page order for coherent reading
        selected.sort(key=lambda r: (r.doc_id, r.page_number or 0, r.chunk_index))

        # 6. Format context + sources
        context_text = self._build_context(selected)
        sources      = self._build_sources(selected)
        total_tokens = sum(r.metadata.get("token_count", 0) for r in selected)

        return RetrievedContext(
            query        = query,
            chunks       = selected,
            context_text = context_text,
            sources      = sources,
            total_tokens = total_tokens,
        )
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    # ── MMR re-ranking ────────────────────────────────────────────────────
    

    def _mmr_rerank(
        self,
        candidates:   List[RetrievalResult],
        query_vector: np.ndarray,
        k:            int,
    ) -> List[RetrievalResult]:
        """
        Maximal Marginal Relevance:
          At each step, pick the chunk that maximises:
            λ * sim(chunk, query) - (1-λ) * max_sim(chunk, already_selected)
          This ensures diversity — avoids selecting 5 near-identical chunks.
        """
        # Pre-compute candidate vectors from score (proxy: we don't re-fetch vectors here)
        # Instead we use a greedy approach with cosine proxy from scores + text similarity
        # Note: for exact MMR you'd store vectors; this is a score-diversity approximation.

        selected:   List[RetrievalResult] = []
        remaining = list(candidates)

        # First pick: highest relevance score
        remaining.sort(key=lambda r: r.score, reverse=True)
        selected.append(remaining.pop(0))

        while len(selected) < k and remaining:
            best_score = -1.0
            best_idx   = 0

            for i, cand in enumerate(remaining):
                # Relevance term
                rel = cand.score

                # Diversity term: penalise if text overlaps with already selected
                similarities = [
                     self._cosine_similarity(cand.vector, sel.vector)
                     for sel in selected
                     if cand.vector is not None and sel.vector is not None
                     ]
                max_sim_to_selected = max(similarities) if similarities else 0.0

                mmr_score = self.mmr_lambda * rel - (1 - self.mmr_lambda) * max_sim_to_selected
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx   = i

            selected.append(remaining.pop(best_idx))

        return selected

    # ── Context builder ───────────────────────────────────────────────────
    
    
    @staticmethod
    def _build_context(chunks: List[RetrievalResult]) -> str:
        """
        Assembles chunks into a structured context string for the LLM prompt.
        Format:
          [Source 1 | doc: report_q3 | page: 4]
          <chunk text>
          ---
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            page_str = f" | page: {chunk.page_number}" if chunk.page_number else ""
            header   = f"[Source {i} | doc: {chunk.doc_id}{page_str} | score: {chunk.score:.2f}]"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_sources(chunks: List[RetrievalResult]) -> List[dict]:
        """Deduplicated list of sources for response citation."""
        seen   = set()
        sources = []
        for chunk in chunks:
            key = (chunk.doc_id, chunk.page_number)
            if key not in seen:
                seen.add(key)
                sources.append({
                    "doc_id":      chunk.doc_id,
                    "source_path": chunk.source_path,
                    "page":        chunk.page_number,
                })
        return sources

    @staticmethod
    def _empty_context(query: str) -> RetrievedContext:
        return RetrievedContext(
            query        = query,
            chunks       = [],
            context_text = "",
            sources      = [],
            total_tokens = 0,
            strategy_used= "no_match",
        )
