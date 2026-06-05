"""Central configuration for the Document Intelligence RAG system."""


CHUNK_SIZE = 1000      
CHUNK_OVERLAP = 150     


EMBEDDING_MODEL = "text-embedding-3-small"  
LLM_MODEL = "gpt-4o-mini"   
LLM_TEMPERATURE = 0.0       

CHROMA_DIR = "chroma_store"   

TOP_K = 4   
