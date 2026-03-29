# backend/app/auth/dependencies.py
from functools import lru_cache
from rag_engine.rag_pipeline.vectorstore.store import build_vector_store, VectorStore
from app.core.config import settings
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
from app.auth.auth_utils import SECRET_KEY, ALGORITHM

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Returns a singleton VectorStore instance, created once and reused."""
    return build_vector_store(
        backend=settings.VECTOR_STORE_BACKEND,
        index_path=settings.FAISS_INDEX_PATH,
        persist_dir=settings.CHROMA_PERSIST_DIR,
    )