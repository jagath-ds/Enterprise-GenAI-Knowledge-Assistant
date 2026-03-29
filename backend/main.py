from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db.database import engine, Base
from app.api.routes.query_routes import router as query_router
from app.api.routes.document_routes import router as document_router
from app.api.routes.auth_routes import router as auth_router
from app.api.routes.admin_routes import router as admin_router
from app.core.rag_pipeline import init_pipeline

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────
    # 1. Create DB tables
    Base.metadata.create_all(bind=engine)
    # 2. Load embedding model + FAISS index + Ollama client — ONCE
    init_pipeline()
    yield
    # ── Shutdown (add cleanup here if needed) ─────────────────────────


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api")
app.include_router(document_router, prefix="/api")
app.include_router(auth_router)
app.include_router(admin_router)


@app.get("/")
async def read_root():
    return {"message": "backend is clean + Azure integrated"}