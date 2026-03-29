from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.query_schema import QueryRequest, QueryResponse
from app.services.rag_service import ask_question
from app.db.database import get_db

router = APIRouter(prefix="/query", tags=["RAG"])

@router.post("/", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    return ask_question(request.question, request.model_mode, db)