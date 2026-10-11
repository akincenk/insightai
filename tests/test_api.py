from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import ai_engine, api
from app.main import app
from app.vector_store import VectorStore
from tests.conftest import fake_embed, make_pdf

API_HEADERS = {"X-API-Key": "test-key"}
client = TestClient(app, headers=API_HEADERS)
anon = TestClient(app)


@pytest.fixture
def mocked_openai(monkeypatch):
    calls = {}

    def fake_answer(question, contexts):
        calls["contexts"] = contexts
        return f"answer to {question}"

    monkeypatch.setattr(api, "store", VectorStore())
    monkeypatch.setattr(ai_engine, "embed", fake_embed)
    monkeypatch.setattr(ai_engine, "answer", fake_answer)
    return calls


def upload(data: bytes, name: str = "doc.pdf", c: TestClient = client):
    return c.post("/upload/", files={"file": (name, data, "application/pdf")})


def test_root():
    assert client.get("/").json() == {"message": "InsightAI is running"}


def test_upload_then_query_returns_most_relevant_chunk(mocked_openai):
    assert upload(make_pdf("python is a snake and a language"), "py.pdf").json()["chunks"] == 1
    assert upload(make_pdf("coffee is a bean drink"), "coffee.pdf").status_code == 200

    resp = client.post("/query/", json={"query": "which drink comes from a bean", "top_k": 1})
    body = resp.json()
    assert resp.status_code == 200
    assert body["answer"] == "answer to which drink comes from a bean"
    assert body["results"][0]["source"] == "coffee.pdf"
    assert mocked_openai["contexts"] == ["coffee is a bean drink"]


def test_query_before_upload_is_rejected(mocked_openai):
    assert client.post("/query/", json={"query": "anything"}).status_code == 400


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 11}, {}])
def test_query_validation(mocked_openai, payload):
    assert client.post("/query/", json=payload).status_code == 422


def test_upload_rejects_non_pdf_extension(mocked_openai):
    assert upload(b"hello", "notes.txt").status_code == 400


def test_upload_rejects_pdf_extension_with_wrong_content(mocked_openai):
    assert upload(b"MZ\x90\x00 not a pdf").status_code == 400


def test_upload_rejects_corrupt_pdf(mocked_openai):
    assert upload(b"%PDF-1.4 garbage").status_code == 400


def test_upload_rejects_oversized_file(mocked_openai, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 100)
    assert upload(make_pdf("x" * 50) + b" " * 200).status_code == 413


def test_upload_rejects_pdf_without_text(mocked_openai):
    assert upload(make_pdf("")).status_code == 422


def test_missing_api_key_returns_503(monkeypatch):
    monkeypatch.setattr(api, "store", VectorStore())
    assert upload(make_pdf("python is a language")).status_code == 503


def test_embed_and_answer_use_v1_client(monkeypatch):
    fake = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda model, input: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0]) for _ in input])
        ),
        responses=SimpleNamespace(create=lambda **kw: SimpleNamespace(output_text=kw["input"])),
    )
    monkeypatch.setattr(ai_engine, "get_client", lambda: fake)
    monkeypatch.setattr(ai_engine, "EMBED_BATCH_SIZE", 2)

    assert ai_engine.embed(["a", "b", "c"]).shape == (3, 2)
    assert "[1] ctx" in ai_engine.answer("q?", ["ctx"])


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_write_endpoints_require_api_key(mocked_openai, headers):
    c = TestClient(app, headers=headers)
    assert upload(make_pdf("python is a language"), c=c).status_code == 401
    assert c.post("/query/", json={"query": "x"}).status_code == 401
    assert len(api.store) == 0


def test_health_check_needs_no_key():
    assert anon.get("/").status_code == 200


def test_allow_no_auth_skips_key_check(mocked_openai, monkeypatch):
    monkeypatch.delenv("API_KEY")
    monkeypatch.setenv("ALLOW_NO_AUTH", "true")
    assert upload(make_pdf("python is a language"), c=anon).status_code == 200


def test_app_refuses_to_start_without_api_key(monkeypatch):
    monkeypatch.delenv("API_KEY")
    with pytest.raises(RuntimeError, match="API_KEY"):
        with TestClient(app):
            pass
    monkeypatch.setenv("ALLOW_NO_AUTH", "true")
    with TestClient(app) as c:
        assert c.get("/").status_code == 200


def test_upload_rejects_too_many_chunks_before_embedding(mocked_openai, monkeypatch):
    monkeypatch.setattr(api, "MAX_CHUNKS_PER_UPLOAD", 1)
    monkeypatch.setattr(ai_engine, "embed", lambda texts: pytest.fail("embed must not be called"))
    long_text = " ".join(f"w{i}" for i in range(300))
    resp = upload(make_pdf(*[long_text[i : i + 80] for i in range(0, len(long_text), 80)]))
    assert resp.status_code == 413
    assert len(api.store) == 0


def test_store_cap_evicts_oldest_document(mocked_openai, monkeypatch):
    monkeypatch.setattr(api, "store", VectorStore(max_chunks=1))
    upload(make_pdf("python is a snake"), "old.pdf")
    upload(make_pdf("coffee is a drink"), "new.pdf")
    body = client.post("/query/", json={"query": "python snake", "top_k": 10}).json()
    assert [r["source"] for r in body["results"]] == ["new.pdf"]
