"""Single shared MiniLM instance. One model, used everywhere."""

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class Embedder:
    def __init__(self):
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        key = text.lower().strip()
        if key not in self._cache:
            self._cache[key] = self._model.encode(
                key, convert_to_numpy=True
            ).astype(np.float32)
        return self._cache[key]

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        keys = [t.lower().strip() for t in texts]
        uncached = [(i, k) for i, k in enumerate(keys) if k not in self._cache]
        if uncached:
            _, new_keys = zip(*uncached)
            vecs = self._model.encode(
                list(new_keys), convert_to_numpy=True, show_progress_bar=False
            ).astype(np.float32)
            for k, v in zip(new_keys, vecs):
                self._cache[k] = v
        return [self._cache[k] for k in keys]
