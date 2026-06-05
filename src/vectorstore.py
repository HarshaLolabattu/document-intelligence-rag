"""ChromaDB vector store + OpenAI embeddings."""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import EMBEDDING_MODEL, CHROMA_DIR, TOP_K


def get_embeddings():
    """Create the OpenAI embedding function."""
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def build_vectorstore(chunks):
    """Embed chunks and store them in ChromaDB. Returns the vector store."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"Stored {len(chunks)} chunk(s) in ChromaDB at '{CHROMA_DIR}'")
    return vectorstore


def load_vectorstore():
    """Load an existing ChromaDB store from disk."""
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )


def get_retriever(vectorstore):
    """Turn the vector store into a retriever that fetches top-K relevant chunks."""
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
