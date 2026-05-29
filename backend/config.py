import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCUMENTS_DIR = os.path.join(BASE_DIR, "Documents")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_DB_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "sws_ai_docs"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
RETRIEVAL_K = 5

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5-coder:3b"

MAX_UPLOAD_SIZE_MB = 20
