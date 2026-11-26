import io
import json

import pytest

from app import ai_engine, api, reranker
from app.vector_store import VectorStore
from tests.conftest import fake_embed, make_pdf
from tests.test_api import client, upload

HITS = [("coffee is a drink", "a.pdf", 1, 0.9), ("python snake", "b.pdf", 2, 0.8), ("tea leaves", "c.pdf", 1, 0.7)]


def fake_urlopen(payload):
    def opener(req, timeout):
        opener.body = json.loads(req.data)
        return io.BytesIO(json.dumps(payload).encode())

    return opener


def test_scores_follow_input_order_and_send_the_query(monkeypatch):
    monkeypatch.setenv("RERANK_URL", "http://127.0.0.1:9/v1/rerank")
    opener = fake_urlopen({"results": [{"index": 2, "relevance_score": 3.0}, {"index": 0, "relevance_score": 1.0},
                                       {"index": 1, "relevance_score": -2.0}]})  # fmt: skip
    monkeypatch.setattr(reranker.urllib.request, "urlopen", opener)
    assert reranker.scores("q", ["a", "b", "c"]) == [1.0, -2.0, 3.0]
    assert opener.body == {"query": "q", "documents": ["a", "b", "c"], "top_n": 3}


def test_rerank_reorders_and_replaces_the_score(monkeypatch):
    monkeypatch.setattr(reranker, "scores", lambda q, docs: [0.1, 5.0, 2.0])
    hits, ok = reranker.rerank("snake", HITS, k=2)
    assert ok and hits == [("python snake", "b.pdf", 2, 5.0), ("tea leaves", "c.pdf", 1, 2.0)]


@pytest.mark.parametrize(
    "payload", [{"results": [{"index": 0, "relevance_score": 1.0}]}, {"oops": []}, {"results": [{"index": 7}]}]
)
def test_a_bad_or_partial_answer_keeps_the_retrieval_order(monkeypatch, payload):
    monkeypatch.setenv("RERANK_URL", "http://127.0.0.1:9/v1/rerank")
    monkeypatch.setattr(reranker.urllib.request, "urlopen", fake_urlopen(payload))
    hits, ok = reranker.rerank("q", HITS, k=2)
    assert not ok and hits == HITS[:2]


def test_an_unreachable_reranker_keeps_the_retrieval_order(monkeypatch):
    monkeypatch.setenv("RERANK_URL", "http://127.0.0.1:9/v1/rerank")

    def down(req, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(reranker.urllib.request, "urlopen", down)
    assert reranker.rerank("q", HITS, k=1) == (HITS[:1], False)


@pytest.fixture
def indexed(monkeypatch):
    monkeypatch.setattr(api, "store", VectorStore())
    monkeypatch.setattr(ai_engine, "embed", fake_embed)
    monkeypatch.setattr(ai_engine, "answer", lambda q, contexts: "ok [1]")
    upload(make_pdf("python is a snake and a language"), "py.pdf")
    upload(make_pdf("coffee is a bean drink"), "coffee.pdf")


def test_query_reranks_when_a_reranker_is_configured(indexed, monkeypatch):
    monkeypatch.setenv("RERANK_URL", "http://127.0.0.1:9/v1/rerank")
    seen = {}

    def fake_scores(q, docs):
        seen["n"] = len(docs)
        return [10.0 if "coffee" in d else 0.0 for d in docs]

    monkeypatch.setattr(reranker, "scores", fake_scores)
    body = client.post("/query/", json={"query": "python snake", "top_k": 1}).json()
    assert body["reranked"] and body["results"][0]["source"] == "coffee.pdf" and seen["n"] == 2
    off = client.post("/query/", json={"query": "python snake", "top_k": 1, "rerank": False}).json()
    assert not off["reranked"] and off["results"][0]["source"] == "py.pdf"


def test_rerank_requested_without_a_reranker_is_an_error(indexed):
    r = client.post("/query/", json={"query": "snake", "rerank": True})
    assert r.status_code == 400 and "RERANK_URL" in r.json()["detail"]
    assert client.post("/query/", json={"query": "snake"}).json()["reranked"] is False
