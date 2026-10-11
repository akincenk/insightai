import numpy as np
import pymupdf
import pytest

VOCAB = ["python", "snake", "language", "coffee", "bean", "drink", "the", "is"]


def fake_embed(texts):
    """Deterministic bag-of-words vectors so retrieval is meaningful without OpenAI."""
    return np.array([[t.lower().split().count(w) + 0.01 for w in VOCAB] for t in texts], dtype=np.float32)


def make_pdf(*pages: str) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture(autouse=True)
def no_real_openai(monkeypatch):
    # Guarantee no test can reach the network even if a key is present locally.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from app import ai_engine

    clear_cache = ai_engine.get_client.cache_clear
    clear_cache()
    yield
    clear_cache()
