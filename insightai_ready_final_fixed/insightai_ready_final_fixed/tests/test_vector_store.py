from app.vector_store import MockVectorStore

def test_vector_store_add_and_search():
    store = MockVectorStore()
    text = "Artificial Intelligence is a branch of computer science."
    store.add_document(text)
    results = store.search("What is AI?")
    assert len(results) > 0
    assert isinstance(results[0][0], str)