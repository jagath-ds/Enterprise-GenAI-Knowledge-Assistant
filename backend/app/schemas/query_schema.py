from pydantic import BaseModel

class Source(BaseModel):
    document_id: str
    source_path: str
    page: int | None

class QueryRequest(BaseModel):
    question: str
    model_mode: str = "auto"

class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]