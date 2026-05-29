# AI-Engineer-RAG-SWS

A Retrieval-Augmented Generation (RAG) chatbot for SWS AI employees to query company policy documents using natural language.

---

## Project Setup

### Prerequisites

- **Python 3.10+** (tested on 3.12)
- **Ollama** with models:
  - `nomic-embed-text` (embeddings)
  - `qwen2.5-coder:3b` (LLM generation — ~3x faster than llama3.1:8b)
- **Pip packages** (see below)

### Installation

```bash
# Activate the Python environment
source ~/Desktop/pyenv/bin/activate

# Install dependencies (if not already present)
pip install fastapi uvicorn chromadb langchain langchain-community \
  langchain-text-splitters pymupdf pdfplumber httpx python-multipart
```

### Run the Server

```bash
# Option 1 — One-command launcher
chmod +x run.sh
./run.sh

# Option 2 — Manual
source ~/Desktop/pyenv/bin/activate
python backend/ingest.py       # Index PDFs into ChromaDB (first run only)
uvicorn backend.main:app --host 0.0.0.0 --port 1234 --reload
```

Open **http://localhost:1234** in your browser.

---

## Build / Run Commands

| Action | Command |
|--------|---------|
| Install dependencies | `pip install -r requirements.txt` |
| Index / re-index documents | `python backend/ingest.py` |
| Start development server | `uvicorn backend.main:app --host 0.0.0.0 --port 1234 --reload` |
| Run with launcher | `./run.sh` |
| Test health endpoint | `curl http://localhost:1234/health` |
| Test chat API | `curl -X POST http://localhost:1234/api/chat -H "Content-Type: application/json" -d '{"question":"What is the leave policy?"}' ` |
| List indexed documents | `curl http://localhost:1234/api/documents` |
| Upload a document | `curl -X POST http://localhost:1234/api/upload -F "file=@/path/to/doc.pdf"` |
| Delete a document | `curl -X DELETE http://localhost:1234/api/documents/SWS-AI-hr-policy.pdf` |

---

## Project Structure

```
SWS/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app — API routes, CORS, static file serving
│   ├── config.py            # Constants — paths, chunk size, model names, ports
│   ├── rag_engine.py        # RAG logic — retrieval, prompt building, LLM query
│   ├── embedding.py         # Custom Ollama embedding wrapper (parallelized)
│   └── ingest.py            # Batch PDF ingestion script (load, chunk, embed, store)
├── frontend/
│   └── index.html           # Single-page chat UI — sidebar, upload modal, notifications
├── Documents/               # 10 company PDFs (HR, leave, IT security, etc.)
├── uploads/                 # User-uploaded PDFs (via the UI)
├── chroma_db/               # Persistent Chroma vector store (auto-generated)
├── run.sh                   # One-command launcher
├── .gitignore
└── README.md
```

---

## Technologies Used

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Backend** | Python 3.12 + FastAPI | REST API server |
| **Vector Database** | ChromaDB (persistent, local) | Store & retrieve document embeddings |
| **Embeddings** | Ollama `nomic-embed-text` (274 MB) | Convert text chunks to vector embeddings |
| **LLM** | Ollama `qwen2.5-coder:3b` (1.9 GB) | Generate grounded answers from retrieved context |
| **PDF Parsing** | PyMuPDF (via LangChain loader) | Extract text from PDF documents |
| **Text Splitting** | LangChain `RecursiveCharacterTextSplitter` | Split documents into 500-char chunks (50 overlap) |
| **Embedding API** | Custom `OllamaEmbeddingFunction` with `ThreadPoolExecutor(8)` | Parallelized embedding for speed |
| **Frontend** | Vanilla HTML / CSS / JavaScript | Chat UI with sidebar, upload modal, notifications |
| **Font** | Google Livvic | UI typography |
| **HTTP** | httpx | Ollama API calls (blocking + streaming) |

### Why These Choices

- **ChromaDB** — Local, zero-config, persistent. No external dependencies.
- **nomic-embed-text** — Lightweight (274 MB), runs entirely on CPU via Ollama.
- **qwen2.5-coder:3b** — Much faster than llama3.1:8b (~3x speedup), sufficient quality for RAG.
- **No LangChain LLM wrappers** — Raw httpx calls to Ollama API to avoid version compatibility issues.
- **Parallel embeddings** — `ThreadPoolExecutor(8)` batches simultaneous embedding requests, reducing ingestion time.
- **Vanilla JS frontend** — Zero build step, no framework overhead, one file to serve.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the chat UI |
| `GET` | `/health` | Health check |
| `GET` | `/api/documents` | List all indexed documents with sizes |
| `POST` | `/api/chat` | Ask a question (returns answer + sources) |
| `POST` | `/api/chat/stream` | Same as chat but with SSE streaming |
| `POST` | `/api/upload` | Upload and index a new PDF document |
| `DELETE` | `/api/documents/{filename}` | Remove a document and its embeddings |

### Chat Request Body

```json
{
  "question": "How many sick days do I get?",
  "documents": ["SWS-AI-leave-policy.pdf", "SWS-AI-hr-policy.pdf"]
}
```

The optional `documents` field restricts retrieval to the listed documents. If omitted, all documents are searched.

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Chunk size | 500 characters | Balances granularity vs. context sufficiency |
| Chunk overlap | 50 characters | Preserves boundary continuity between chunks |
| Retrieval top-k | 5 | Covers enough context without noise |
| LLM temperature | 0.1 | Low creativity — ensures faithful grounding |
| Embedding parallelism | 8 threads | Near-linear speedup on modern CPUs |
| LLM `num_predict` | 1024 tokens | Sufficient for detailed policy answers |

---

## Sample Test Queries

- "What is the annual leave policy at SWS AI?"
- "How many days of sick leave do employees get?"
- "What is the notice period for resignation?"
- "What tools does SWS AI use for communication?"
- "What is the password policy for company systems?"
- "How are performance reviews conducted?"
- "What are the WFH guidelines?"
- "Does SWS AI offer health insurance?"

---

## Assumptions & Notes

1. **Ollama must be running** on `localhost:11434` with the required models pulled (`nomic-embed-text`, `qwen2.5-coder:3b`).
2. **Python environment** uses the pyenv at `~/Desktop/pyenv/` (run `source ~/Desktop/pyenv/bin/activate` to activate).
3. **Uploaded documents** are stored in `uploads/` and added to the same ChromaDB collection. Documents in `Documents/` are not modifiable via the UI (they are the base company policies).
4. **Removing a document** deletes both the file and its embeddings from ChromaDB. Re-upload the document to re-index it.
5. **Select/deselect** checkboxes in the sidebar filter which documents the RAG engine searches. Only affects the current chat session.
6. **Answer grounding** — The LLM is explicitly instructed to answer *only* from the provided context. If the answer is not in the documents, it responds: "I don't have that information in the company documents."
7. **PDF parsing** uses PyMuPDF. Complex layouts (tables, multi-column) may not parse perfectly — text extraction quality depends on the PDF structure.
