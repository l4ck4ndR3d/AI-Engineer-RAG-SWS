import httpx
from langchain_core.embeddings import Embeddings

from config import OLLAMA_BASE_URL, EMBED_MODEL


class OllamaEmbeddingFunction(Embeddings):
    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = httpx.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("embeddings", [])

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]
