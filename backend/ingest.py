import os
import glob

from config import DOCUMENTS_DIR, CHROMA_DB_DIR, COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, OLLAMA_BASE_URL, EMBED_MODEL

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from embedding import OllamaEmbeddingFunction


def load_pdfs(documents_dir: str):
    pdf_files = glob.glob(os.path.join(documents_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in {documents_dir}")
    all_docs = []
    for pdf_path in sorted(pdf_files):
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(pdf_path)
        all_docs.extend(docs)
    return all_docs


def ingest_documents():
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    raw_docs = load_pdfs(DOCUMENTS_DIR)
    print(f"Loaded {len(raw_docs)} raw document pages")

    chunks = text_splitter.split_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    embeddings = OllamaEmbeddingFunction(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
    )

    print(f"Ingested {len(chunks)} chunks into ChromaDB at {CHROMA_DB_DIR}")
    return vectorstore


if __name__ == "__main__":
    ingest_documents()
