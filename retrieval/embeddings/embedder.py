from typing import List
from retrieval.embeddings.model import EmbeddingModel


class Embedder:
    def __init__(self):
        self._model = None

    def _get_model(self):
        if self._model is None:
            self._model = EmbeddingModel.get_model()
        return self._model

    def encode(self, text: str) -> List[float]:
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty.")

        print("Embedder: getting model", flush=True)

        model = self._get_model()

        print("Embedder: model obtained", flush=True)

        print("Embedder: starting encode", flush=True)

        embedding = model.encode(
            text,
            normalize_embeddings=True,
        )

        print("Embedder: encoding finished", flush=True)

        return embedding.tolist()

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("Input text list cannot be empty.")

        embeddings = self._get_model().encode(
            texts,
            normalize_embeddings=True,
        )

        return embeddings.tolist()