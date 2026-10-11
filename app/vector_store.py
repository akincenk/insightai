import numpy as np
import hashlib
from typing import List, Tuple

def mock_embed(text: str, dim: int = 1536) -> np.ndarray:
    hash_digest = hashlib.sha256(text.encode()).digest()
    np.random.seed(int.from_bytes(hash_digest[:4], 'big'))
    return np.random.rand(dim).astype('float32')

class MockVectorStore:
    def __init__(self, dim: int = 1536):
        self.dim = dim
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []

    def add_document(self, text: str):
        emb = mock_embed(text, self.dim)
        self.documents.append(text)
        self.embeddings.append(emb)

    def search(self, query: str, k: int = 1) -> List[Tuple[str, float]]:
        query_vec = mock_embed(query, self.dim)
        sims = [self.cosine_similarity(query_vec, emb) for emb in self.embeddings]
        sorted_indices = np.argsort(sims)[::-1][:k]
        return [(self.documents[i], sims[i]) for i in sorted_indices]

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))