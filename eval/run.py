"""Retrieval evaluation: keyword (BM25) vs vector vs hybrid (RRF) on a fictional handbook.

Usage:
  python -m eval.run                  # embeddings from the configured OpenAI-compatible endpoint
  OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=local \\
  OPENAI_EMBEDDING_MODEL=embeddinggemma python -m eval.run   # fully local with Ollama
  python -m eval.run --fake           # offline smoke test (bag-of-words vectors, not a real score)
  python -m eval.run --set codes      # the identifier-heavy set (error codes, part numbers)
  python -m eval.run --rerank         # also each mode with its 20 best candidates reranked via RERANK_URL
"""

import argparse
import json
from pathlib import Path

from app import reranker
from app.pdf_utils import chunk_pages
from app.vector_store import VectorStore
from eval.metrics import mrr, recall_at

HERE = Path(__file__).parent
MODES = ("keyword", "vector", "hybrid")
SETS = {"handbook": ("corpus.json", "questions.json"), "codes": ("corpus_codes.json", "questions_codes.json")}


def build(embed, corpus: str = "corpus.json"):
    store = VectorStore()
    for doc in json.loads((HERE / corpus).read_text())["documents"]:
        paged = chunk_pages(doc["pages"])
        texts = [c for c, _ in paged]
        store.add(texts, embed(texts), source=doc["source"], pages=[n for _, n in paged])
    return store


def evaluate(embed, k_max: int = 10, name: str = "handbook", rerank: bool = False) -> dict[str, dict[str, float]]:
    corpus, qfile = SETS[name]
    store = build(embed, corpus)
    questions = json.loads((HERE / qfile).read_text())
    vectors = embed([q["q"] for q in questions])
    report = {}
    for mode in [*MODES, *(f"{m}+rerank" for m in MODES if rerank)]:
        ranks = []
        for q, vec in zip(questions, vectors, strict=True):
            base = mode.removesuffix("+rerank")
            hits = store.hybrid_search(q["q"], vec, k=20 if mode != base else k_max, mode=base)
            if mode != base:
                hits, ok = reranker.rerank(q["q"], hits, k_max)
                if not ok:
                    raise RuntimeError("reranking failed; check RERANK_URL")
            gold = [i for i, (_, src, page, _) in enumerate(hits, 1) if (src, page) == (q["source"], q["page"])]
            ranks.append(gold[0] if gold else None)
        report[mode] = {
            "recall@1": recall_at(ranks, 1),
            "recall@3": recall_at(ranks, 3),
            "recall@5": recall_at(ranks, 5),
            "MRR": mrr(ranks),
        }
    return report


def table(report) -> str:
    rows = ["| Mode | Recall@1 | Recall@3 | Recall@5 | MRR |", "| --- | --- | --- | --- | --- |"]
    rows += [
        f"| {m} | {r['recall@1']:.2f} | {r['recall@3']:.2f} | {r['recall@5']:.2f} | {r['MRR']:.2f} |"
        for m, r in report.items()
    ]
    return "\n".join(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--set", default="handbook", choices=sorted(SETS))
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    if args.fake:
        from tests.conftest import fake_embed as embed
    else:
        from app.ai_engine import embed
    print(table(evaluate(embed, name=args.set, rerank=args.rerank)))
