"""Answer evaluation: are the generated answers right, are their citations right, and does the service admit
when the documents do not say?

For each of the 30 handbook questions the answer must contain every gold fact (eval/answers.json; one
alternative per group, case-insensitive), cite at least one excerpt, and its citations should point to the page
that holds the answer. For 10 questions the handbook does not answer, the reply must say it could not find the
answer (the instruction in app/ai_engine.py) instead of inventing one.

Grading is lexical, so it is lenient where a fact is a plain yes or no; citation precision and abstention are
the stricter signals.

Usage (fully local with Ollama):
  OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=local OPENAI_EMBEDDING_MODEL=embeddinggemma2-bf16 \\
  OPENAI_CHAT_MODEL=qwen3.8:27b python -m eval.answers [--rerank] [--json out.json]
"""

import argparse
import json
import re
from pathlib import Path

from app import ai_engine, reranker
from eval.run import build

HERE = Path(__file__).parent
ABSTAIN = ("could not find", "couldn't find", "cannot find", "can't find", "not found", "no information",
           "do not contain", "don't contain", "does not contain", "doesn't contain", "not mention",
           "not in the documents", "not covered", "does not say", "doesn't say")  # fmt: skip


def cited(text: str) -> list[int]:
    return [int(n) for n in re.findall(r"\[(\d+)\]", text)]


def abstained(text: str) -> bool:
    t = text.lower().replace("’", "'")
    return any(p in t for p in ABSTAIN)


def has_facts(text: str, facts: list[list[str]]) -> bool:
    t = text.lower()
    return all(any(alt.lower() in t for alt in group) for group in facts)


def grade(item: dict | None, text: str, hits: list[tuple]) -> dict:
    """item None = an unanswerable question."""
    refs = cited(text)
    if item is None:
        return {"abstained": abstained(text), "citations": len(refs)}
    gold = [i for i, h in enumerate(hits, 1) if (h[1], h[2]) == (item["source"], item["page"])]
    good = [n for n in refs if n in gold]
    return {
        "correct": has_facts(text, item["facts"]),
        "cited": bool(refs),
        "citations": len(refs),
        "gold_citations": len(good),
        "retrieved_gold": bool(gold),
        "abstained": abstained(text),
    }


def evaluate(embed, answer, k: int = 3, mode: str = "vector", rerank: bool = False, candidates: int = 20) -> dict:
    store = build(embed)
    data = json.loads((HERE / "answers.json").read_text())
    items = [(it, it["q"]) for it in data["answerable"]] + [(None, q) for q in data["unanswerable"]]
    rows = []
    for item, q in items:
        hits = store.hybrid_search(q, embed([q])[0], k=candidates if rerank else k, mode=mode)
        if rerank:
            hits, _ = reranker.rerank(q, hits, k)
        text = answer(q, [h[0] for h in hits])
        rows.append({"q": q, "answer": text, **grade(item, text, hits)})
        print(json.dumps({"q": q[:50], **{x: y for x, y in rows[-1].items() if x not in ("q", "answer")}}), flush=True)
    ans = [r for r in rows if "correct" in r]
    una = [r for r in rows if "correct" not in r]
    n_cit = sum(r["citations"] for r in ans)
    summary = {
        "correct": f"{sum(r['correct'] for r in ans)}/{len(ans)}",
        "cited": f"{sum(r['cited'] for r in ans)}/{len(ans)}",
        "citation_precision": round(sum(r["gold_citations"] for r in ans) / n_cit, 3) if n_cit else None,
        "gold_page_retrieved": f"{sum(r['retrieved_gold'] for r in ans)}/{len(ans)}",
        "false_abstentions": f"{sum(r['abstained'] for r in ans)}/{len(ans)}",
        "abstained_when_unanswerable": f"{sum(r['abstained'] for r in una)}/{len(una)}",
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--mode", default="vector", choices=["vector", "keyword", "hybrid"])
    p.add_argument("--rerank", action="store_true", help="rerank 20 candidates through RERANK_URL")
    p.add_argument("--json")
    a = p.parse_args()
    out = evaluate(ai_engine.embed, ai_engine.answer, a.k, a.mode, a.rerank)
    print(json.dumps(out["summary"], indent=2))
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
