from app.pdf_utils import chunk_pages
from app.vector_store import VectorStore, bm25_rank, rrf
from eval.metrics import mrr, recall_at
from tests.conftest import fake_embed


def test_chunk_pages_keeps_page_numbers_and_skips_empty_pages():
    chunks = chunk_pages(["alpha beta", "", "gamma"], size=200, overlap=40)
    assert chunks == [("alpha beta", 1), ("gamma", 3)]


def test_bm25_finds_exact_codes_that_embeddings_blur():
    texts = [
        "fill in form RX-17 for damaged pallets",
        "the returns desk opens at nine",
        "pallets are stacked by height",
    ]
    assert bm25_rank("RX-17", texts)[0] == 0


def test_rrf_rewards_items_both_rankings_agree_on():
    assert set(rrf([[2, 0, 1], [0, 2, 1]])[:2]) == {0, 2}
    assert rrf([[1, 0], [1, 2]])[0] == 1


def test_hybrid_search_returns_source_page_and_modes_differ():
    store = VectorStore()
    texts = ["python is a snake", "coffee is a bean drink", "error code E-42 means the language server stopped"]
    store.add(texts, fake_embed(texts), source="doc.pdf", pages=[1, 2, 3])
    hits = store.hybrid_search("E-42", fake_embed(["E-42"])[0], k=2)
    assert hits[0][:3] == (texts[2], "doc.pdf", 3)
    vec = store.hybrid_search("snake", fake_embed(["snake"])[0], k=1, mode="vector")
    assert vec[0][2] == 1


def test_metrics():
    ranks = [1, 2, None, 1]  # rank of the gold chunk per question, None = not found
    assert recall_at(ranks, 1) == 0.5 and recall_at(ranks, 3) == 0.75
    assert abs(mrr(ranks) - (1 + 0.5 + 0 + 1) / 4) < 1e-9


def test_keyword_mode_returns_nothing_when_no_word_matches():
    store = VectorStore()
    store.add(["coffee is a bean drink"], fake_embed(["coffee is a bean drink"]), source="d.pdf", pages=[1])
    assert store.hybrid_search("zebra", fake_embed(["zebra"])[0], mode="keyword") == []


def test_eval_harness_runs_offline_and_scores_every_mode():
    from eval.run import MODES, evaluate, table

    report = evaluate(fake_embed)
    assert set(report) == set(MODES)
    assert all(0 <= r["MRR"] <= 1 for r in report.values())
    assert report["keyword"]["MRR"] > 0  # exact codes such as RX-17 must be found by BM25
    assert table(report).startswith("| Mode |")


def test_common_words_never_count_as_keyword_matches():
    # "can I ... a ... the" must not make every chunk a keyword hit and push the vector winner out
    texts = ["the dock opens at six", "a pallet can be stacked", "customers may return unused items within 30 days"]
    assert bm25_rank("Can I give back a product?", texts) == []
