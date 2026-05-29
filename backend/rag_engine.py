import httpx

from config import CHROMA_DB_DIR, COLLECTION_NAME, RETRIEVAL_K, OLLAMA_BASE_URL, EMBED_MODEL, LLM_MODEL
from embedding import OllamaEmbeddingFunction
from langchain_community.vectorstores import Chroma


SYSTEM_PROMPT = (
    "You are a helpful HR assistant for SWS AI. Answer the employee's question "
    "using ONLY the provided context from the company documents. "
    "If the answer is not contained in the context, say: "
    "'I don't have that information in the company documents.' "
    "Be concise and accurate. Cite the source document name when possible."
)


class RAGEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddingFunction(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
        self.vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=CHROMA_DB_DIR,
        )

    def retrieve(self, question: str):
        results = self.vectorstore.similarity_search_with_score(question, k=RETRIEVAL_K)
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

    def query_ollama_llm(self, prompt: str) -> str:
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},
        }
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json().get("response", "").strip()

    def query(self, question: str):
        results = self.retrieve(question)
        context, sources = self.format_context(results)

        full_prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {question}

Answer:"""

        answer = self.query_ollama_llm(full_prompt)
        return {"answer": answer, "sources": sources}
