import os
import sys
import uuid
import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from config import DOCUMENTS_DIR, UPLOADS_DIR, CHROMA_DB_DIR, COLLECTION_NAME, MAX_UPLOAD_SIZE_MB
from rag_engine import RAGEngine
from embedding import OllamaEmbeddingFunction
from ingest import chunk_documents
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

os.makedirs(UPLOADS_DIR, exist_ok=True)

app = FastAPI(title="SWS AI RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = RAGEngine()


class ChatRequest(BaseModel):
    question: str
    documents: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class DocItem(BaseModel):
    name: str
    size: int


def get_all_documents():
    docs = []
    seen = set()
    for f in glob.glob(os.path.join(DOCUMENTS_DIR, "*.pdf")):
        name = os.path.basename(f)
        if name not in seen:
            seen.add(name)
            docs.append(DocItem(name=name, size=os.path.getsize(f)))
    for f in glob.glob(os.path.join(UPLOADS_DIR, "*.pdf")):
        name = os.path.basename(f)
        if name not in seen:
            seen.add(name)
            docs.append(DocItem(name=name, size=os.path.getsize(f)))
    return docs


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents():
    return {"documents": get_all_documents()}


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_UPLOAD_SIZE_MB}MB limit")

    filepath = os.path.join(UPLOADS_DIR, file.filename)
    if os.path.exists(filepath):
        base, ext = os.path.splitext(file.filename)
        filepath = os.path.join(UPLOADS_DIR, f"{base}_{uuid.uuid4().hex[:8]}{ext}")

    with open(filepath, "wb") as f:
        f.write(content)

    loader = PyMuPDFLoader(filepath)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = os.path.basename(filepath)

    chunks, ids = chunk_documents(docs)

    embeddings = OllamaEmbeddingFunction()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR,
    )
    vectorstore.add_documents(documents=chunks, ids=ids)
    vectorstore.persist()

    return {
        "message": f"Ingested {len(chunks)} chunks from {file.filename}",
        "filename": os.path.basename(filepath),
        "chunks": len(chunks),
    }


@app.delete("/api/documents/{filename:path}")
def delete_document(filename: str):
    in_documents = os.path.exists(os.path.join(DOCUMENTS_DIR, filename))
    if in_documents:
        raise HTTPException(status_code=403, detail="Cannot remove base company documents. Only user-uploaded documents can be deleted.")

    filepath = os.path.join(UPLOADS_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")

    os.remove(filepath)

    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=OllamaEmbeddingFunction(),
            persist_directory=CHROMA_DB_DIR,
        )
        all_docs_in_store = vectorstore.get()
        ids_to_delete = []
        for i, meta in enumerate(all_docs_in_store.get("metadatas", [])):
            if meta and meta.get("source") == filename:
                ids_to_delete.append(all_docs_in_store["ids"][i])
        if ids_to_delete:
            vectorstore.delete(ids_to_delete)
        vectorstore.persist()
    except Exception as e:
        pass

    return {"message": f"Document '{filename}' removed", "filename": filename}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.question.strip():
        return ChatResponse(answer="Please ask a question.", sources=[])
    result = rag.query(request.question, doc_filter=request.documents)
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@app.post("/api/chat/stream")
def chat_stream(request: ChatRequest):
    if not request.question.strip():
        return StreamingResponse(
            iter([f"data: {ChatResponse(answer='Please ask a question.', sources=[]).model_dump_json()}\n\n"]),
            media_type="text/event-stream",
        )
    return StreamingResponse(
        rag.query_stream(request.question, doc_filter=request.documents),
        media_type="text/event-stream",
    )


@app.get("/")
def serve_ui():
    return FileResponse(str(FRONTEND_DIR / "index.html"))
