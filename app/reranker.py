"""Optional second-stage reranking with a cross-encoder.

Retrieval scores the question and each chunk separately; a cross-encoder reads them together, which ranks
better but is too slow for every chunk, so it only reorders the first candidates. Any server with the common
`/v1/rerank` shape works (llama.cpp's llama-server with `--reranking`, vLLM, Jina, Cohere-compatible gateways).

The endpoint comes only from the environment (RERANK_URL), never from a request, so a caller cannot point the
server at another address.
"""

import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)
TIMEOUT = float(os.getenv("RERANK_TIMEOUT", "30"))


class RerankError(RuntimeError):
    pass


def configured() -> bool:
    return bool(os.getenv("RERANK_URL"))


def scores(query: str, documents: list[str]) -> list[float]:
    """One relevance score per document, in input order. Raises RerankError on any failure."""
    url = os.getenv("RERANK_URL")
    if not url:
        raise RerankError("RERANK_URL is not set")
    body = {"query": query, "documents": documents, "top_n": len(documents)}
    if model := os.getenv("RERANK_MODEL"):
        body["model"] = model
    headers = {"Content-Type": "application/json"}
    if key := os.getenv("RERANK_API_KEY"):
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            results = json.load(r)["results"]
        out = [float("-inf")] * len(documents)
        for item in results:
            out[int(item["index"])] = float(item["relevance_score"])
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise RerankError(f"rerank failed: {type(exc).__name__}") from exc
    if float("-inf") in out:
        raise RerankError("rerank answer did not score every document")
    return out


def rerank(query: str, hits: list[tuple], k: int) -> tuple[list[tuple], bool]:
    """Reorder (text, source, page, score) hits by cross-encoder score; keep the input order if reranking fails.

    Returns (top k hits, whether they were reranked). The score in each hit becomes the reranker's score.
    """
    if not hits:
        return [], False
    try:
        s = scores(query, [h[0] for h in hits])
    except RerankError as exc:
        log.warning("%s; using the retrieval order", exc)
        return hits[:k], False
    order = sorted(range(len(hits)), key=lambda i: -s[i])
    return [(*hits[i][:3], s[i]) for i in order[:k]], True
