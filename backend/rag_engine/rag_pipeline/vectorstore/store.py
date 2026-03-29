from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

from rag_engine.rag_pipeline.embeddings.bge_engine import EmbeddedChunk
EMBEDDING_DIM = 1024

logger = logging.getLogger(__name__)


# ─── Result type ─────────────────────────────────────────────────────────────

class RetrievalResult:
    """One retrieved chunk with its similarity score."""
    __slots__ = ("chunk_id", "text", "doc_id", "source_path", "page_number",
                 "chunk_index", "score", "metadata","vector")

    def __init__(self, *, chunk_id, text, doc_id, source_path,
                 page_number, chunk_index, score, metadata=None, vector=None):
        self.chunk_id    = chunk_id
        self.text        = text
        self.doc_id      = doc_id
        self.source_path = source_path
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.score       = score       # cosine similarity (0–1)
        self.metadata    = metadata or {}
        self.vector      = vector      

    def __repr__(self):
        return (f"RetrievalResult(doc={self.doc_id!r}, page={self.page_number}, "
                f"score={self.score:.3f}, text={self.text[:60]!r}…)")


# ─── Abstract base ───────────────────────────────────────────────────────────

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, embedded_chunks: List[EmbeddedChunk]) -> None: ...

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None,
    ) -> List[RetrievalResult]: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def persist(self) -> None: ...


# ─── ChromaDB store ──────────────────────────────────────────────────────────

class ChromaVectorStore(VectorStore):
    """
    Persistent ChromaDB-backed store.
    Each chunk is stored with its full metadata so you can inspect/filter
    without re-loading the source documents.
    """

    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_chunks"):
        import chromadb
        from chromadb.config import Settings

        self._client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},   # use cosine distance
        )
        logger.info(f"ChromaDB ready: {persist_dir}/{collection_name}  "
                    f"({self._collection.count()} existing chunks)")

    def upsert(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        if not embedded_chunks:
            return

        ids        = [ec.chunk.chunk_id for ec in embedded_chunks]
        embeddings = [ec.vector.tolist() for ec in embedded_chunks]
        documents  = [ec.chunk.text for ec in embedded_chunks]
        metadatas  = [
            {
                "doc_id":      ec.chunk.doc_id,
                "source_path": ec.chunk.source_path,
                "page_number": ec.chunk.page_number or -1,
                "chunk_index": ec.chunk.chunk_index,
                "total_chunks": ec.chunk.total_chunks,
                "token_count": ec.chunk.token_count,
                **{k: str(v) for k, v in ec.chunk.metadata.items()},
            }
            for ec in embedded_chunks
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas, #type: ignore
        )
        logger.debug(f"Upserted {len(embedded_chunks)} chunks into ChromaDB.")

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_doc_id: Optional[str] = None,
    ) -> List[RetrievalResult]:
        where = {"doc_id": filter_doc_id} if filter_doc_id else None #type: ignore

        results = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self._collection.count() or 1),
            where=where, #type: ignore
            include=["documents", "metadatas", "distances"],
        )
        if not results or not results.get("documents"):
            return []

        retrieval = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            # Chroma returns cosine DISTANCE (0=identical, 2=opposite)
            # Convert to similarity: 1 - dist/2  →  range [0, 1]
            score = 1.0 - dist / 2.0
            retrieval.append(RetrievalResult(
                chunk_id    = results["ids"][0][i],
                text        = doc,
                doc_id      = meta.get("doc_id", ""),
                source_path = meta.get("source_path", ""),
                page_number = meta.get("page_number"),
                chunk_index = meta.get("chunk_index", 0),
                score       = score,
                metadata    = meta,
            ))
        return retrieval

    def delete_document(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})
        logger.info(f"Deleted all chunks for doc_id='{doc_id}'")

    def count(self) -> int:
        return self._collection.count()

    def persist(self) -> None:
        # ChromaDB PersistentClient auto-persists; explicit flush here
        logger.info("ChromaDB auto-persists; no explicit flush needed.")


# ─── FAISS store (fallback) ───────────────────────────────────────────────────

class FAISSVectorStore(VectorStore):
    """
    Persistent FAISS vector store using IndexIDMap2 wrapping IndexFlatIP.

    Why IndexIDMap2?
      - Maps arbitrary integer IDs → vectors (enables targeted deletion)
      - Inner-product search = cosine similarity on L2-normalized vectors (fast)
      - Persists to disk: index.faiss + metadata.json sidecar
      - Full upsert/delete/reload cycle without re-embedding

    ID mapping: chunk_id (str) ↔ int64 via a stable hash
    """

    def __init__(self, index_path: str = "./faiss_index", **kwargs):
        import faiss
        self._faiss      = faiss
        self._index_path = Path(index_path)
        self._index_path.mkdir(parents=True, exist_ok=True)

        self._index_file = self._index_path / "index.faiss"
        self._meta_file  = self._index_path / "metadata.json"

        # chunk_id (str) → metadata dict
        self._metadata: Dict[str, dict] = {}
        # chunk_id (str) → int64 numeric ID used in FAISS
        self._id_to_int: Dict[str, int] = {}
        self._int_to_id: Dict[int, str] = {}

        if self._index_file.exists():
            self._index = faiss.read_index(str(self._index_file))
            with open(self._meta_file) as f:
                saved           = json.load(f)
                self._metadata  = saved["metadata"]
                self._id_to_int = {k: int(v) for k, v in saved["id_to_int"].items()}
                self._int_to_id = {int(k): v for k, v in saved["int_to_id"].items()}
            logger.info(f"Loaded FAISS index ({self._index.ntotal} vectors) ← {index_path}")
        else:
            flat         = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._index  = faiss.IndexIDMap2(flat)   # wraps flat with ID mapping
            logger.info(f"Created new FAISS IndexIDMap2(IndexFlatIP) at {index_path}")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _chunk_id_to_int(self, chunk_id: str) -> int:
        """Stable int64 hash of a chunk_id string for FAISS ID mapping."""
        import hashlib
        h = hashlib.sha256(chunk_id.encode()).digest()
        # Take first 8 bytes as unsigned int, then make it fit in int64
        return int.from_bytes(h[:8], "big") >> 1  # >> 1 keeps it positive int63

    # ── VectorStore interface ─────────────────────────────────────────────

    def upsert(self, embedded_chunks: List[EmbeddedChunk]) -> None:
        new_vecs = []
        new_ids  = []

        for ec in embedded_chunks:
            cid     = ec.chunk.chunk_id
            int_id  = self._chunk_id_to_int(cid)

            # Remove existing vector first if re-indexing same chunk
            if cid in self._id_to_int:
                try:
                    self._index.remove_ids(
                        self._faiss.IDSelectorBatch(
                            1, self._faiss.swig_ptr(
                                np.array([int_id], dtype=np.int64)
                            )
                        )
                    )
                except Exception:
                    pass  # not critical if removal fails

            self._id_to_int[cid]    = int_id
            self._int_to_id[int_id] = cid
            self._metadata[cid]     = {
                "text":        ec.chunk.text,
                "doc_id":      ec.chunk.doc_id,
                "source_path": ec.chunk.source_path,
                "page_number": ec.chunk.page_number,
                "chunk_index": ec.chunk.chunk_index,
                "token_count": ec.chunk.token_count,
                **ec.chunk.metadata,
            }
            vec = ec.vector.astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            new_vecs.append(vec)
            new_ids.append(int_id)

        if new_vecs:
            vecs_np = np.stack(new_vecs)
            ids_np  = np.array(new_ids, dtype=np.int64)
            self._index.add_with_ids(vecs_np, ids_np)

        logger.debug(f"FAISS index size after upsert: {self._index.ntotal}")

    def search(
        self,
        query_vector:  np.ndarray,
        top_k:         int            = 5,
        filter_doc_id: Optional[str]  = None,
    ) -> List[RetrievalResult]:
        if self._index.ntotal == 0:
            return []

        # Fetch more if filtering so we still return top_k after filter
        fetch_k = min(top_k * 4 if filter_doc_id else top_k, self._index.ntotal)
        q = query_vector.reshape(1, -1).astype(np.float32)
        q=q / np.linalg.norm(q)

        scores, int_ids = self._index.search(q, fetch_k)
        results: List[RetrievalResult] = []

        for score, int_id in zip(scores[0], int_ids[0]):
            if int_id < 0:
                continue
            cid  = self._int_to_id.get(int_id)
            if cid is None:
                continue
            meta = self._metadata[cid]
            if not meta:
                continue
            if filter_doc_id and meta.get("doc_id") != filter_doc_id:
                continue

            vector = None

            results.append(RetrievalResult(
                chunk_id    = cid,
                text        = meta["text"],
                doc_id      = meta.get("doc_id", ""),
                source_path = meta.get("source_path", ""),
                page_number = meta.get("page_number"),
                chunk_index = meta.get("chunk_index", 0),
                score       = float(score),
                metadata    = meta,
                vector      = vector,
            ))
            if len(results) >= top_k:
                break

        return results

    def delete_document(self, doc_id: str) -> None:
        """Remove all vectors belonging to a document using IndexIDMap2's remove_ids."""
        target_int_ids = [
            self._id_to_int[cid]
            for cid, meta in self._metadata.items()
            if meta.get("doc_id") == doc_id and cid in self._id_to_int
        ]
        if not target_int_ids:
            logger.warning(f"delete_document: no chunks found for doc_id='{doc_id}'")
            return

        ids_np = np.array(target_int_ids, dtype=np.int64)
        selector = self._faiss.IDSelectorBatch(len(ids_np), self._faiss.swig_ptr(ids_np))
        removed  = self._index.remove_ids(selector)

        # Clean up metadata maps
        to_remove = [cid for cid, meta in self._metadata.items()
                     if meta.get("doc_id") == doc_id]
        for cid in to_remove:
            int_id = self._id_to_int.pop(cid, None)
            if int_id is not None:
                self._int_to_id.pop(int_id, None)
            del self._metadata[cid]

        logger.info(f"Deleted {removed} vectors for doc_id='{doc_id}'")

    def count(self) -> int:
        return self._index.ntotal

    def persist(self) -> None:
        self._faiss.write_index(self._index, str(self._index_file))
        with open(self._meta_file, "w") as f:
            json.dump({
                "metadata":  self._metadata,
                "id_to_int": self._id_to_int,
                "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
            }, f)
        logger.info(f"FAISS index persisted → {self._index_path} ({self._index.ntotal} vectors)")


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_vector_store(backend: str = "chroma", **kwargs) -> VectorStore:
    """
    Factory to select backend.
    backend: 'chroma' | 'faiss'
    """
    if backend == "chroma":
        return ChromaVectorStore(**kwargs)
    elif backend == "faiss":
        return FAISSVectorStore(**kwargs)
    else:
        raise ValueError(f"Unknown vector store backend: {backend!r}. Use 'chroma' or 'faiss'.")
