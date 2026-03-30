from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Dict

from rag_engine.rag_pipeline.chunking.semantic_chunker import SemanticChunker
from rag_engine.rag_pipeline.embeddings.bge_engine import EmbeddingEngine
from rag_engine.rag_pipeline.vectorstore.store import build_vector_store, VectorStore
from rag_engine.rag_pipeline.retrieval.retrieval_engine import RetrievalEngine
from rag_engine.rag_pipeline.llm.orchestrator import LLMOrchestrator, LLMResponse

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # Chunking
    chunk_size_tokens:    int   = 400
    chunk_overlap_tokens: int   = 80

    # Embeddings
    embedding_model:      str   = "BAAI/bge-large-en-v1.5"
    embedding_batch_size: int   = 32
    embedding_cache:      bool  = True

    # Vector store
    vector_backend:       str   = "faiss"
    vector_persist_dir:   str   = "./data/vector_store"
    collection_name:      str   = "enterprise_rag"

    # Retrieval
    # fast model uses retrieval_top_k_fast, deep model uses retrieval_top_k_deep
    # caller's explicit top_k argument always overrides both
    retrieval_top_k_fast: int   = 3
    retrieval_top_k_deep: int   = 6
    retrieval_fetch_k:    int   = 25
    min_relevance_score:  float = 0.30
    mmr_lambda:           float = 0.6

    # LLM
    fast_model:           str   = "phi3:mini"
    deep_model:           str   = "mistral:latest"
    llm_backend:          str   = "ollama"
    llm_base_url:         str   = "http://localhost:11434"
    llm_temperature:      float = 0.1
    llm_max_tokens:       int   = 800


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Public API
    ----------
    index_document(...)   — chunk → embed → store
    index_pages(...)      — multi-page variant
    delete_document(...)  — remove by doc_id
    query(...)            — embed → retrieve → LLM → LLMResponse
    stats()               — diagnostics
    """

    def __init__(
        self,
        chunker:              SemanticChunker,
        embedder:             EmbeddingEngine,
        store:                VectorStore,
        retriever:            RetrievalEngine,
        llm:                  LLMOrchestrator,
        retrieval_top_k_fast: int = 3,
        retrieval_top_k_deep: int = 6,
    ):
        self.chunker              = chunker
        self.embedder             = embedder
        self.store                = store
        self.retriever            = retriever
        self.llm                  = llm
        self.retrieval_top_k_fast = retrieval_top_k_fast
        self.retrieval_top_k_deep = retrieval_top_k_deep
        logger.info("RAGPipeline initialised.")

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: Optional[PipelineConfig] = None) -> "RAGPipeline":
        if cfg is None:
            cfg = PipelineConfig()

        logger.info("Building RAGPipeline from config…")
        os.makedirs(cfg.vector_persist_dir, exist_ok=True)

        chunker = SemanticChunker(
            chunk_size = cfg.chunk_size_tokens,
            overlap    = cfg.chunk_overlap_tokens,
        )
        embedder = EmbeddingEngine(
            model_name = cfg.embedding_model,
            batch_size = cfg.embedding_batch_size,
            cache      = cfg.embedding_cache,
        )
        # ✅ Fixed
        if cfg.vector_backend == "faiss":
            store = build_vector_store(
                backend    = "faiss",
                index_path = cfg.vector_persist_dir,  # FAISS expects index_path not persist_dir
            )
        elif cfg.vector_backend == "chroma":
            store = build_vector_store(
                backend         = "chroma",
                persist_dir     = cfg.vector_persist_dir,
                collection_name = cfg.collection_name,
            )
        else:
            raise ValueError(f"Unknown vector backend: {cfg.vector_backend!r}")
        retriever = RetrievalEngine(
            embedding_engine = embedder,
            vector_store     = store,
            top_k            = cfg.retrieval_top_k_deep,   # default; overridden per query
            fetch_k          = cfg.retrieval_fetch_k,
            min_score        = cfg.min_relevance_score,
            mmr_lambda       = cfg.mmr_lambda,
        )
        llm = LLMOrchestrator(
            fast_model  = cfg.fast_model,
            deep_model  = cfg.deep_model,
            backend     = cfg.llm_backend,
            base_url    = cfg.llm_base_url,
            temperature = cfg.llm_temperature,
            max_tokens  = cfg.llm_max_tokens,
        )
        return cls(
            chunker,
            embedder,
            store,
            retriever,
            llm,
            retrieval_top_k_fast = cfg.retrieval_top_k_fast,
            retrieval_top_k_deep = cfg.retrieval_top_k_deep,
        )

    # ── Indexing ──────────────────────────────────────────────────────────

    def index_document(
        self,
        text:        str,
        doc_id:      str,
        source_path: str,
        page_number: Optional[int]  = None,
        extra_meta:  Optional[dict] = None,
    ) -> int:
        """
        Full indexing pipeline for a single document / page:
          text → chunks → embeddings → vector store
        Returns number of chunks indexed.
        """
        logger.info(f"Indexing doc_id={doc_id!r}  source={source_path!r}")

        chunks = self.chunker.chunk_document(
            text        = text,
            doc_id      = doc_id,
            source_path = source_path,
            page_number = page_number,
            extra_meta  = extra_meta or {},
        )
        if not chunks:
            logger.warning(f"No chunks produced for doc_id={doc_id!r}. Skipping.")
            return 0

        avg_tokens = sum(c.token_count for c in chunks) // len(chunks)
        logger.info(f"  → {len(chunks)} chunks (avg {avg_tokens} tokens each)")

        embedded = self.embedder.embed_chunks(chunks)
        self.store.upsert(embedded)
        logger.info(
            f"  → Indexed {len(embedded)} vectors. "
            f"Store total: {self.store.count()}"
        )
        return len(chunks)

    def index_pages(
        self,
        pages:       List[Dict],    # [{"text": ..., "page": 1}, ...]
        doc_id:      str,
        source_path: str,
        extra_meta:  Optional[dict] = None,
    ) -> int:
        """Index a multi-page document preserving page numbers."""
        chunks   = self.chunker.chunk_pages(pages, doc_id, source_path, extra_meta or {})
        embedded = self.embedder.embed_chunks(chunks)
        self.store.upsert(embedded)
        logger.info(f"Indexed {len(embedded)} chunks for multi-page doc {doc_id!r}")
        return len(embedded)

    def delete_document(self, doc_id: str) -> None:
        """Remove all chunks for a document (useful before re-indexing)."""
        self.store.delete_document(doc_id)
        logger.info(f"Deleted all chunks for doc_id={doc_id!r}")

    # ── Querying ──────────────────────────────────────────────────────────

    def query(
        self,
        question:      str,
        model_mode:    str           = "auto",   # 'auto' | 'fast' | 'deep'
        top_k:         Optional[int] = None,
        filter_doc_id: Optional[str] = None,
    ) -> LLMResponse:
        """
        Full RAG query pipeline:
          question → select model → retrieve context → LLM → LLMResponse

        Model selection happens ONCE here.
        Retrieval depth scales automatically with model:
          fast → retrieval_top_k_fast (default 3)
          deep → retrieval_top_k_deep (default 6)
        Caller's explicit top_k always overrides both.
        """
        logger.info(f"Query: {question!r}  |  mode={model_mode!r}")

        # 1. Select model — single call, no internal override
        self.llm.select_model(question, model_mode)
        logger.info(f"  → Active model: {self.llm.model!r}")

        # 2. Resolve retrieval depth
        if top_k is not None:
            effective_top_k = top_k
        elif self.llm.model == self.llm.fast_model:
            effective_top_k = self.retrieval_top_k_fast
        else:
            effective_top_k = self.retrieval_top_k_deep

        logger.info(f"  → Retrieval top_k: {effective_top_k}")

        # 3. Retrieve context
        context = self.retriever.retrieve(
            query         = question,
            top_k         = effective_top_k,
            filter_doc_id = filter_doc_id,
        )
        logger.info(
            f"  → Retrieved {len(context.chunks)} chunks "
            f"(~{context.total_tokens} tokens)"
        )

        # 4. Generate answer (metadata cleaning happens inside orchestrator)
        response = self.llm.answer(context)
        logger.info(
            f"  → {'Grounded' if response.is_grounded else 'FALLBACK'} answer | "
            f"model={response.model_used!r}"
        )

        return response

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "total_indexed_chunks":  self.store.count(),
            "embedding_cache":       self.embedder.cache_stats(),
            "active_model":          self.llm.model,
            "fast_model":            self.llm.fast_model,
            "deep_model":            self.llm.deep_model,
            "retrieval_top_k_fast":  self.retrieval_top_k_fast,
            "retrieval_top_k_deep":  self.retrieval_top_k_deep,
        }