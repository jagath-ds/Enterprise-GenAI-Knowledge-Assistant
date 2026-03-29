from pydantic import BaseModel
from uuid import UUID
from enum import Enum


class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    indexing = "indexing"
    indexed = "indexed"
    failed = "failed"

class DocumentUploadResponse(BaseModel):
    document_id: UUID
    file_name: str
    status: DocumentStatus

class IndexRequest(BaseModel):
    document_id: UUID

class IndexResponse(BaseModel):
    message: str
    document_id: str

