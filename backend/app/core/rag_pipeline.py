from __future__ import annotations

import logging
from typing import Optional
from app.core.config import settings
from rag_engine.rag_pipeline.pipeline import RAGPipeline, PipelineConfig

logger = logging.getLogger(__name__)

_pipeline: Optional[RAGPipeline] = None


def init_pipeline() -> None:
    global _pipeline
    if _pipeline is not None:
        logger.info("Pipeline already initialised — skipping rebuild.")
        return

    logger.info("Initialising RAGPipeline singleton…")
    cfg = PipelineConfig(
        vector_backend     = settings.VECTOR_STORE_BACKEND,  
        vector_persist_dir = settings.FAISS_INDEX_PATH,     
    )
    _pipeline = RAGPipeline.from_config(cfg)
    logger.info("RAGPipeline singleton ready.")


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError(
            "Pipeline not initialised. "
            "Ensure init_pipeline() is called in FastAPI startup."
        )
    return _pipeline