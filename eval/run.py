"""Retrieval evaluation: keyword (BM25) vs vector vs hybrid (RRF) on a fictional handbook.

Usage:
  python -m eval.run                  # embeddings from the configured OpenAI-compatible endpoint
  OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=local \\
  OPENAI_EMBEDDING_MODEL=embeddinggemma python -m eval.run   # fully local with Ollama
  python -m eval.run --fake           # offline smoke test (bag-of-words vectors, not a real score)
"""

import json
import sys
from pathlib import Path

from app.pdf_utils import chunk_pages
from app.vector_store import VectorStore
from eval.metrics import mrr, recall_at

HERE = Path(__file__).parent
MODES = ("keyword", "vector", "hybrid")


def build(embed):
    store = VectorStore()
    for doc in json.loads((HERE / "corpus.json").read_text())["documents"]:
        paged = chunk_pages(doc["pages"])
        texts = [c for c, _ in paged]
        store.add(texts, embed(texts), source=doc["source"], pages=[n for _, n in paged])
    return store


def evaluate(embed, k_max: int = 10) -> dict[str, dict[str, float]]:
    store = build(embed)
    questions = json.loads((HERE / "questions.json").read_text())
    vectors = embed([q["q"] for q in questions])
    report = {}
    for mode in MODES:
        ranks = []
        for q, vec in zip(questions, vectors, strict=True):
            hits = store.hybrid_search(q["q"], vec, k=k_max, mode=mode)
            gold = [i for i, (_, src, page, _) in enumerate(hits, 1) if (src, page) == (q["source"], q["page"])]
            ranks.append(gold[0] if gold else None)
        report[mode] = {"recall@1": recall_at(ranks, 1), "recall@3": recall_at(ranks, 3),
                        "recall@5": recall_at(ranks, 5), "MRR": mrr(ranks)}
    return report


def table(report) -> str:
    rows = ["| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |", "| --- | --- | --- | --- | --- |"]
    rows += [f"| {m} | {r['recall@1']:.2f} | {r['recall@3']:.2f} | {r['recall@5']:.2f} | {r['MRR']:.2f} |"
             for m, r in report.items()]
    return "\n".join(rows)


if __name__ == "__main__":
    if "--fake" in sys.argv:
        from tests.conftest import fake_embed as embed
    else:
        from app.ai_engine import embed
    print(table(evaluate(embed)))
