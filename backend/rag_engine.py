import json
import httpx

from config import CHROMA_DB_DIR, COLLECTION_NAME, RETRIEVAL_K, OLLAMA_BASE_URL, EMBED_MODEL, LLM_MODEL
from embedding import OllamaEmbeddingFunction
from langchain_community.vectorstores import Chroma


SYSTEM_PROMPT = (
    "You are a helpful HR assistant for SWS AI. Answer the employee's question "
    "using ONLY the provided context from the company documents. "
    "If the answer is not contained in the context, say: "
    "'I don't have that information in the company documents.' "
    "Give a thorough, well-structured answer. Use bullet points where appropriate. "
    "Always cite the source document name at the end of your answer. "
    "Format your response in clear paragraphs."
)


class RAGEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddingFunction(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR,
        )

    def retrieve(self, question: str, doc_filter: list[str] | None = None):
        where_filter = None
        if doc_filter:
            where_filter = {"source": {"$in": doc_filter}}
        results = self.vectorstore.similarity_search_with_score(
            question, k=RETRIEVAL_K, filter=where_filter
        )
        return results

    def format_context(self, results):
        context_parts = []
        sources = []
        seen = set()
        for doc, score in results:
            context_parts.append(f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content}")
            src = doc.metadata.get("source", "Unknown")
            if src not in seen:
                seen.add(src)
                sources.append(src)
        return "\n\n".join(context_parts), sources

    def query(self, question: str, doc_filter: list[str] | None = None):
        results = self.retrieve(question, doc_filter)
        context, sources = self.format_context(results)

        if not context.strip():
            return {"answer": "I don't have that information in the company documents.", "sources": []}

        full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""

        payload = {
            "model": LLM_MODEL,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        answer = response.json().get("response", "").strip()
        return {"answer": answer, "sources": sources}

    def query_stream(self, question: str, doc_filter: list[str] | None = None):
        results = self.retrieve(question, doc_filter)
        context, sources = self.format_context(results)

        if not context.strip():
            yield f"data: {json.dumps({'token': "I don't have that information in the company documents."})}\n\n"
            yield f"data: {json.dumps({'sources': []})}\n\n"
            return

        full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""

        payload = {
            "model": LLM_MODEL,
            "prompt": full_prompt,
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": 1024},
        }
        with httpx.Client() as client:
            with client.stream(
                "POST", f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=120,
            ) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield f"data: {json.dumps({'token': token})}\n\n"
                        if data.get("done", False):
                            yield f"data: {json.dumps({'sources': sources})}\n\n"
                            break
                    except json.JSONDecodeError:
                        continue
