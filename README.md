# AI-Engineer-RAG-SWS

SWS AI internal policy RAG chatbot — employees ask natural language questions and get grounded answers from company documents.

## Architecture

- **Backend**: Python (FastAPI) — `backend/main.py`
- **Ingestion**: `backend/ingest.py` — loads 10 PDFs, splits into 500-char chunks (50 overlap), embeds via `nomic-embed-text`, stores in ChromaDB
- **Retrieval**: ChromaDB (local, persistent) — similarity search, top-k=5
- **Generation**: Ollama `llama3.1:8b` — answers only from retrieved context
- **Frontend**: Vanilla HTML/CSS/JS — white + blue theme, Livvic font, source display
- **Embeddings**: Ollama `nomic-embed-text` via custom `OllamaEmbeddingFunction`

## Quick Start

```bash
chmod +x run.sh
./run.sh
```

This activates the pyenv, ingests PDFs (first run only), and starts the server at `http://localhost:1234`.

Or manually:

```bash
source ~/Desktop/pyenv/bin/activate
python backend/ingest.py          # ingest PDFs into ChromaDB
uvicorn backend.main:app --host 0.0.0.0 --port 1234 --reload
```

## Design Decisions

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Vector DB | Chroma | Local, persistent, zero-config |
| Embedding | nomic-embed-text | Lightweight, runs locally via Ollama |
| LLM | llama3.1:8b | Strong instruction-following, good RAG performance |
| Chunking | 500 chars / 50 overlap | Balances granularity and context |
| Retrieval k | 5 | Covers enough context without noise |
| Framework | FastAPI + raw Ollama API | Lightweight, minimal dependency issues |

## Test Queries

- "What is the annual leave policy at SWS AI?"
- "How many days of sick leave do employees get?"
- "What is the notice period for resignation?"
- "What tools does SWS AI use for communication?"
- "What is the password policy for company systems?"
- "How are performance reviews conducted?"
- "What are the WFH guidelines?"
- "Does SWS AI offer health insurance?"
# AI-Engineer-RAG-SWS
