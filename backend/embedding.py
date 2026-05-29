from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
from langchain_core.embeddings import Embeddings

from config import OLLAMA_BASE_URL, EMBED_MODEL


class OllamaEmbeddingFunction(Embeddings):
    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def _embed_single(self, text: str) -> list[float]:
        response = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(self._embed_single, t): i for i, t in enumerate(texts)}
            results = [None] * len(texts)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_single(text)
