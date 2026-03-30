from fastapi import APIRouter, File, UploadFile, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
import uuid
from app.db.database import get_db
from app.services.document_service import delete_document, prepare_indexing
from app.services.rag_service import perform_indexing
from app.services.azure_storage import AzureStorageService
from app.db.models import Document
from app.schemas.document_schema import DocumentStatus
from app.utils.logger import log_event
azure_service = AzureStorageService()

router = APIRouter(prefix="/documents", tags=["Documents"])


# ✅ UPLOAD
@router.post("/upload")
def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="invalid filename")
          # ── Duplicate check ──
        existing = db.query(Document).filter(Document.file_name == file.filename).first()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"File '{file.filename}' already exists"
            )
        blob_name = azure_service.upload_file(file)

        new_doc = Document(
            document_id=uuid.uuid4(),
            file_name=file.filename,
            file_path=blob_name,
            status=DocumentStatus.uploaded.value
        )

        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        log_event(
            db,
            actor_type="admin",
            actor_id=1,
            action_type="UPLOAD_DOCUMENT",
            description=f"Uploaded {file.filename}"
        )
        

        prepare_indexing(new_doc.document_id, db)

        background_tasks.add_task(perform_indexing, new_doc.document_id, True)

        return {
            "document_id": str(new_doc.document_id),
            "file_name": new_doc.file_name,
            "status": "uploaded + indexing started"
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


# ✅ GET DOCUMENTS 
@router.get("/")
def get_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()

    return [
        {
            "id": str(doc.document_id),
            "filename": doc.file_name,
            "status": doc.status
        }
        for doc in docs
    ]


# ✅ INDEX
@router.post("/{document_id}/index")
def index_document(
    document_id: uuid.UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    force: bool = False
):
    try:
        doc_id = prepare_indexing(document_id, db, force)
        background_tasks.add_task(perform_indexing, doc_id,force)

        return {"message": "Indexing started", "document_id": str(doc_id)}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ✅ DELETE
@router.delete("/{document_id}")
def delete_document_api(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),  # vector_store removed entirely
):
    try:
        return delete_document(document_id=document_id, db=db)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))