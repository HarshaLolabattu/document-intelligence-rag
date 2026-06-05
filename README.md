# Document Intelligence System using RAG and LLMs

A Retrieval Augmented Generation (RAG) pipeline that enables semantic
question-answering over custom document collections (PDF and text). Built with
LangChain and OpenAI GPT, using ChromaDB for vector storage and configurable
chunking for context optimization. Document ingestion and query endpoints are
exposed via FastAPI, with responses grounded in retrieved context to reduce
hallucination on domain-specific queries.

## Features

- Semantic Q&A over custom PDF and text documents
- LangChain + OpenAI GPT RAG pipeline
- ChromaDB vector storage with OpenAI embeddings
- Configurable chunking (chunk size + overlap) for context optimization
- FastAPI endpoints for document ingestion and querying
- Grounded responses that reduce hallucination by answering only from retrieved context

## Tech Stack

Python, LangChain, OpenAI API, ChromaDB, FastAPI, Retrieval Augmented Generation (RAG)

## Architecture
