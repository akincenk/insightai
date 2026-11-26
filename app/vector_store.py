import math
import re
import threading
from collections import Counter

import numpy as np

_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
RRF_K = 60  # standard reciprocal-rank-fusion constant
CANDIDATES = 50  # per ranking, before fusion


# Function words carry no topic; counting them as keyword hits pushes good vector matches out of the fusion.
STOP = frozenset(
    """a about after all am an and any are as at be been before by can could did do does for from get had
    has have how i if in into is it its may me might must my no not of on or our shall should so some
    than that the their them then there these they this to too us was we were what when where which who
    why will with would you your""".split()
)


def tokens(text: str) -> list[str]:
    """Lower-case content words; hyphenated codes such as RX-17 stay one token."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def bm25_rank(query: str, texts: list[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of `texts` ordered by Okapi BM25 score; texts sharing no query word are left out."""
    q = set(tokens(query))
    docs = [tokens(t) for t in texts]
    if not q or not docs:
        return []
    avg = sum(map(len, docs)) / len(docs) or 1
    df = Counter(w for d in docs for w in set(d) if w in q)
    scores = []
    for i, d in enumerate(docs):
        tf = Counter(w for w in d if w in q)
        s = sum(math.log(1 + (len(docs) - df[w] + 0.5) / (df[w] + 0.5)) * f * (k1 + 1)
                / (f + k1 * (1 - b + b * len(d) / avg)) for w, f in tf.items())
        if s > 0:
            scores.append((s, i))
    return [i for _, i in sorted(scores, reverse=True)]


def rrf(rankings: list[list[int]]) -> list[int]:
    """Reciprocal rank fusion: items ranked high by several rankings win."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            score[item] = score.get(item, 0.0) + 1 / (RRF_K + pos + 1)
    return sorted(score, key=score.get, reverse=True)


class VectorStore:
    """In-memory hybrid search: cosine similarity over normalised embeddings plus BM25 keywords.

    Search is brute force, which is fast enough for the few thousand chunks the store is capped at.
    When more than `max_chunks` are stored, the oldest chunks are evicted first.
    """

    def __init__(self, max_chunks: int | None = None) -> None:
        if max_chunks is not None and max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = max_chunks
        self._lock = threading.Lock()
        self._texts: list[str] = []
        self._sources: list[str] = []
        self._pages: list[int | None] = []
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._texts)

    def add(self, texts: list[str], embeddings: np.ndarray, source: str, pages: list[int] | None = None) -> None:
        if len(texts) != len(embeddings) or (pages is not None and len(pages) != len(texts)):
            raise ValueError("texts, embeddings and pages length mismatch")
        normed = _normalise(np.asarray(embeddings, dtype=np.float32))
        with self._lock:
            self._matrix = normed if self._matrix is None else np.vstack([self._matrix, normed])
            self._texts.extend(texts)
            self._sources.extend([source] * len(texts))
            self._pages.extend(pages if pages is not None else [None] * len(texts))
            overflow = len(self._texts) - self.max_chunks if self.max_chunks else 0
            if overflow > 0:
                self._matrix = self._matrix[overflow:]
                del self._texts[:overflow], self._sources[:overflow], self._pages[:overflow]

    def search(self, query_vec: np.ndarray, k: int = 3) -> list[tuple[str, str, float]]:
        with self._lock:
            if self._matrix is None:
                return []
            scores = self._matrix @ _normalise(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
            top = np.argsort(scores)[::-1][:k]
            return [(self._texts[i], self._sources[i], float(scores[i])) for i in top]

    def hybrid_search(self, query: str, query_vec: np.ndarray, k: int = 3,
                      mode: str = "hybrid") -> list[tuple[str, str, int | None, float]]:
        """(text, source, page, score) best first. mode: "hybrid" (default), "vector" or "keyword"."""
        if mode not in ("hybrid", "vector", "keyword"):
            raise ValueError("mode must be hybrid, vector or keyword")
        with self._lock:
            if self._matrix is None:
                return []
            sims = self._matrix @ _normalise(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))[0]
            vector = [int(i) for i in np.argsort(sims)[::-1][:CANDIDATES]]
            keyword = bm25_rank(query, self._texts)[:CANDIDATES] if mode != "vector" else []
            order = vector if mode == "vector" else keyword if mode == "keyword" else rrf([vector, keyword])
            return [(self._texts[i], self._sources[i], self._pages[i], float(sims[i])) for i in order[:k]]


def _normalise(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1, norms)
