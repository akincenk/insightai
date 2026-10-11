import threading

import numpy as np


class VectorStore:
    """In-memory cosine-similarity search over normalised embeddings.

    Search is a brute-force matrix product, which is fast enough for the few thousand
    chunks the store is capped at. When more than `max_chunks` are stored, the oldest
    chunks are evicted first.
    """

    def __init__(self, max_chunks: int | None = None) -> None:
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = max_chunks
        self._lock = threading.Lock()
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, texts: list[str], embeddings: np.ndarray, source: str) -> None:
        if len(texts) != len(embeddings):
            raise ValueError("texts and embeddings length mismatch")
        normed = _normalise(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            self._matrix = normed if self._matrix is None else np.vstack([self._matrix, normed])
            self._texts.extend(texts)
            self._sources.extend([source] * len(texts))
            overflow = len(self._texts) - self.max_chunks if self.max_chunks else 0
            if overflow > 0:
                self._matrix = self._matrix[overflow:]
                del self._texts[:overflow], self._sources[:overflow]

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[tuple[str, str, float]]:
        with self._lock:
            if self._matrix is None:
                return []
            scores = self._matrix @ _normalise(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
            top = np.argsort(scores)[::-1][:k]
            return [(self._texts[i], self._sources[i], float(scores[i])) for i in top]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
