"""Retrieval metrics over the 1-based rank of the gold chunk per question (None = not retrieved)."""


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr(ranks: list[int | None]) -> float:
    return sum(1 / r for r in ranks if r) / len(ranks)
