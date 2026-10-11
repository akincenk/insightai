import numpy as np
import pytest

from app.vector_store import VectorStore


def test_empty_store_returns_no_results():
    assert VectorStore().search(np.ones(3)) == []


def test_search_ranks_by_cosine_similarity():
    store = VectorStore()
    store.add(["x-axis", "y-axis", "diagonal"], np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]]), source="a.pdf")
    results = store.search(np.array([0.9, 0.1, 0]), k=2)
    assert [r[0] for r in results] == ["x-axis", "diagonal"]
    assert results[0][1] == "a.pdf"
    assert results[0][2] == pytest.approx(0.9 / np.hypot(0.9, 0.1))


def test_k_larger_than_store_and_multiple_sources():
    store = VectorStore()
    store.add(["one"], np.array([[1.0, 0.0]]), source="a.pdf")
    store.add(["two"], np.array([[0.0, 1.0]]), source="b.pdf")
    results = store.search(np.array([0.0, 1.0]), k=10)
    assert len(store) == 2
    assert [(t, s) for t, s, _ in results] == [("two", "b.pdf"), ("one", "a.pdf")]


def test_zero_vector_does_not_produce_nan():
    store = VectorStore()
    store.add(["zero"], np.zeros((1, 2)), source="a.pdf")
    assert store.search(np.array([1.0, 0.0]))[0][2] == 0.0


def test_length_mismatch_rejected():
    with pytest.raises(ValueError):
        VectorStore().add(["a", "b"], np.ones((1, 2)), source="a.pdf")
