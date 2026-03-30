from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy.orm import Session

from rag_engine.rag_pipeline.document_loader import DocumentLoader
from app.core.rag_pipeline import get_pipeline
from app.db.models import Document
from app.db.database import sessionLocal
from app.schemas.document_schema import DocumentStatus
from app.schemas.query_schema import QueryResponse, Source
from app.services.azure_storage import AzureStorageService

import tempfile

azure_service = AzureStorageService()


# ─── Query ────────────────────────────────────────────────────────────────────

def ask_question(question: str, model_mode: str, db: Session) -> QueryResponse:

    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    rag = get_pipeline()   # always instant — singleton already built at startup

    response = rag.query(
        question   = question,
        model_mode = model_mode,
    )

    seen    = set()
    sources = []

    for s in response.sources:
        key = (s["source_path"], s["page"])
        if key in seen:
            continue
        seen.add(key)

        try:
            doc_uuid = uuid.UUID(s["doc_id"])
            document = db.get(Document, doc_uuid)
            display_name = document.file_name if document else Path(s["source_path"]).name
        except (ValueError, AttributeError):
            display_name = Path(s["source_path"]).name

        sources.append(
            Source(
                document_id = s["doc_id"],
                source_path = display_name,
                page        = s["page"],
            )
        )

    return QueryResponse(
        answer   = response.answer,
        grounded = response.is_grounded,
        sources  = sources,
    )


# ─── Document indexing ────────────────────────────────────────────────────────

def perform_indexing(document_id: uuid.UUID, force: bool = False) -> None:

    db  = sessionLocal()
    rag = get_pipeline()
    document = None  # guard so the except block can always reference it safely

    try:
        document = db.query(Document).filter(
            Document.document_id == document_id
        ).first()

        if document is None:
            raise HTTPException(status_code=404, detail="Document not found.")

        if document.status in (DocumentStatus.indexing, DocumentStatus.indexed) and not force:
            raise HTTPException(
                status_code=400,
                detail="Document is already indexed. Use force=true to re-index.",
            )

        document.status = DocumentStatus.indexing
        db.commit()

        file_bytes = azure_service.download_file(document.file_path)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / document.file_name
            temp_path.write_bytes(file_bytes)

            loader    = DocumentLoader(docs_dir=temp_dir, recursive=False)
            documents = loader.load_all()

            if not documents:
                document.status = DocumentStatus.failed
                db.commit()
                raise HTTPException(
                    status_code=400,
                    detail="No content found in document.",
                )

            if force:
                rag.delete_document(str(document_id))
                rag.store.persist()

            for doc in documents:
                current_document = db.get(Document, document_id)
                if current_document is None:
                    rag.delete_document(str(document_id))
                    rag.store.persist()
                    return

                rag.index_pages(
                    pages       = doc["pages"],
                    doc_id      = str(document_id),
                    source_path = doc["source_path"],
                )

            current_document = db.get(Document, document_id)
            if current_document is None:
                rag.delete_document(str(document_id))
                rag.store.persist()
                return

            rag.store.persist()

        document.status = DocumentStatus.indexed
        db.commit()

    except HTTPException:
        raise

    except Exception as e:
        # Only update status if the document row was actually fetched
        if document is not None:
            document.status = DocumentStatus.failed
            db.commit()
        raise HTTPException(
            status_code=500,
            detail=f"Indexing failed: {e}",
        )

    finally:
        db.close()


# ─── Text indexing ────────────────────────────────────────────────────────────

def index_text(text: str, doc_id: str, source_path: str) -> dict:

    rag           = get_pipeline()
    chunks_created = rag.index_document(
        text        = text,
        doc_id      = doc_id,
        source_path = source_path,
    )
    rag.store.persist()

    return {
        "status":         "indexed",
        "chunks_created": chunks_created,
    }
