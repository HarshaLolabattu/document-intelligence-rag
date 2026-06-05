"""Load documents (PDF + text) and split them into chunks."""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_document(file_path: str):
    """Load a single file. Supports .pdf and .txt."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .txt")

    return loader.load()


def chunk_documents(documents):
    """Split documents into overlapping chunks for context optimization."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_file(file_path: str):
    """Full pipeline for one file: load -> chunk. Returns list of chunks."""
    docs = load_document(file_path)
    chunks = chunk_documents(docs)
    print(f"Loaded '{file_path}' -> {len(docs)} page(s) -> {len(chunks)} chunk(s)")
    return chunks
