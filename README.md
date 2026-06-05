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

## Setup

1. Clone the repository:
```bash
   git clone https://github.com/HarshaLolabattu/document-intelligence-rag.git
   cd document-intelligence-rag
```

2. Install dependencies:
```bash
   pip install -r requirements.txt
```

3. Configure your OpenAI API key:
```bash
   cp .env.example .env
   # edit .env and add your real OPENAI_API_KEY
```

## Usage

Run the FastAPI server:
```bash
uvicorn src.api:app --reload
```

Then open the interactive docs at http://localhost:8000/docs

### Endpoints

- `GET /` - health check
- `POST /ingest` - upload a PDF or TXT file to ingest into the vector store
- `POST /query` - ask a question and receive an answer grounded in the documents

Example query request:
```json
{
  "question": "What is RAG and how does it reduce hallucination?"
}
```

## Author

Harsha Adinaraynaraju
Email: harshaadinarayana@gmail.com
