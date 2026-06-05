"""FastAPI app exposing document ingestion and query endpoints."""

import os
import shutil
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from src.ingest import ingest_file
from src.vectorstore import build_vectorstore, load_vectorstore, get_retriever
from src.rag_chain import build_rag_chain, answer_question
from src.config import CHROMA_DIR

app = FastAPI(title="Document Intelligence RAG API")

# Holds the active RAG chain in memory after documents are ingested
state = {"chain": None}

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""
    question: str


@app.get("/")
def health():
    """Simple health check."""
    return {"status": "ok", "message": "Document Intelligence RAG API is running"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF or TXT file, chunk it, and store it in ChromaDB."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    chunks = ingest_file(file_path)
    vectorstore = build_vectorstore(chunks)
    retriever = get_retriever(vectorstore)
    state["chain"] = build_rag_chain(retriever)

    return {
        "filename": file.filename,
        "chunks_stored": len(chunks),
        "message": "Document ingested. You can now query it.",
    }


@app.post("/query")
def query(request: QueryRequest):
    """Ask a question against the ingested documents."""
    if state["chain"] is None:
        if os.path.exists(CHROMA_DIR):
            retriever = get_retriever(load_vectorstore())
            state["chain"] = build_rag_chain(retriever)
        else:
            return {"error": "No documents ingested yet. Use /ingest first."}

    answer = answer_question(state["chain"], request.question)
    return {"question": request.question, "answer": answer}
