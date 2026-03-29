from __future__ import annotations

import logging
import os
from typing import Optional

from rag_engine.rag_pipeline.pipeline import RAGPipeline, PipelineConfig

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_DIR = os.path.join(BASE_DIR, "..", "data", "vector_store")

_pipeline: Optional[RAGPipeline] = None


def init_pipeline() -> None:
    """
    Build and cache the RAGPipeline exactly once.
    Must be called from FastAPI startup — never per-request.
    """
    global _pipeline
    if _pipeline is not None:
        logger.info("Pipeline already initialised — skipping rebuild.")
        return

    logger.info("Initialising RAGPipeline singleton…")
    cfg = PipelineConfig(
        vector_backend      = "faiss",
        vector_persist_dir  = VECTOR_DIR,
        llm_backend         = "ollama",
    )
    _pipeline = RAGPipeline.from_config(cfg)
    logger.info("RAGPipeline singleton ready.")


def get_pipeline() -> RAGPipeline:
    """
    Returns the cached singleton. Raises if init_pipeline() was never called.
    """
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialised. "
            "Ensure init_pipeline() is called in FastAPI startup."
        )
    return _pipeline