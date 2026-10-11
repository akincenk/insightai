import threading

import numpy as np


class VectorStore:
    """In-memory cosine-similarity search over normalised embeddings."""

    # ponytail: brute-force numpy scan held in process memory; swap for FAISS/pgvector if corpora grow large
    def __init__(self) -> None:
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
