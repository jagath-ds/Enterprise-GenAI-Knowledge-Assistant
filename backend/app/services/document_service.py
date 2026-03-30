import shutil
import uuid
from pathlib import Path
from typing import BinaryIO
from sqlalchemy.orm import Session
from werkzeug.utils import secure_filename
from app.services.azure_storage import AzureStorageService
from app.db.models import Document
from app.schemas.document_schema import DocumentStatus, DocumentUploadResponse
from app.core.config import settings
from app.utils.logger import log_event
from app.core.rag_pipeline import get_pipeline

azure_service = AzureStorageService()

EXTENSIONS = {".pdf", ".docx", ".txt",".pptx",".html",".htm"}


def upload_document(filename: str, file_stream: BinaryIO, db: Session) -> DocumentUploadResponse:

    filename = secure_filename(filename)

    if not filename :
        raise ValueError("No file uploaded")

    file_extension = Path(filename).suffix.lower()
    if file_extension not in EXTENSIONS:
        raise ValueError("File extension not allowed")
    
    document_id = uuid.uuid4()
    document_folder = Path(settings.STORAGE_PATH) / str(document_id)

    try:
        document_folder.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise ValueError("Storage conflict detected")
    
    file_path = document_folder / filename
    relative_file_path = str(Path(str(document_id)) / filename)

    try:
        with open(file_path,"wb") as buffer:
            shutil.copyfileobj(file_stream,buffer)
    
    except Exception as e:
        if document_folder.exists():
            shutil.rmtree(document_folder)
        raise ValueError("Failed to save the file")

    new_document = Document(
        document_id=document_id,
        file_name=filename,
        file_path=relative_file_path,
        status= DocumentStatus.uploaded.value
    )
    try:
        db.add(new_document)
        db.commit()
        db.refresh(new_document)
    except Exception as e:
        db.rollback()
        if document_folder.exists():
            shutil.rmtree(document_folder)
        raise ValueError("Failed to save the document")
    return DocumentUploadResponse(
            document_id=new_document.document_id,
            file_name=new_document.file_name,
            status= DocumentStatus(new_document.status)
            )

def prepare_indexing(document_id:uuid.UUID, db: Session, force:bool = False):
    document = db.get(Document, document_id)
    if not document:
        raise ValueError("Document not found")

    if document.status == DocumentStatus.indexing.value:
        raise ValueError("Document is already being indexed")
    if document.status == DocumentStatus.indexed.value and not force:
        raise ValueError("Document is already indexed. Use force=true to re-index")    

    document.status = DocumentStatus.indexing.value
    document.error_message = None
    db.commit()

    return document.document_id

def delete_document(document_id: uuid.UUID, db: Session):
    
    document = db.get(Document, document_id)

    if not document:
        raise ValueError("Document not found")

    # 🔥 Step 1: Delete from Azure

    blob_name = document.file_path   

    if blob_name:
        azure_service.delete_file(blob_name)

    # Step 2: Delete from vector store via pipeline singleton
    rag = get_pipeline()
    rag.delete_document(str(document_id))
    rag.store.persist()

    # 🔥 Step 3: Delete from DB
    log_event(
        db,
        actor_type="admin",
        actor_id=1,
        action_type="DELETE_DOCUMENT",
        description=f"Deleted {document.file_name}"
    )
    db.delete(document)
    db.commit()
    return {"status": "deleted", "document_id": str(document_id)}