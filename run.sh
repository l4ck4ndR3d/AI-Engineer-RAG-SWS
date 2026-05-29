#!/usr/bin/env bash
source /home/cyborg/Desktop/pyenv/bin/activate

echo "=== SWS AI RAG Chatbot ==="

if [ ! -d "/home/cyborg/Documents/SWS/chroma_db" ] || [ -z "$(ls -A /home/cyborg/Documents/SWS/chroma_db 2>/dev/null)" ]; then
    echo "Ingesting documents into vector database..."
    python /home/cyborg/Documents/SWS/backend/ingest.py
    echo "Ingestion complete!"
else
    echo "Vector database already exists."
fi

echo "Starting FastAPI server on http://0.0.0.0:1234"
echo "Open the chat UI at http://localhost:1234"
uvicorn backend.main:app --host 0.0.0.0 --port 1234 --reload
