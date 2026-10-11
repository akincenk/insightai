import pytest

from app.pdf_utils import InvalidPDFError, chunk_text, extract_text
from tests.conftest import make_pdf


def test_extract_text_reads_all_pages_and_normalises_whitespace():
    text = extract_text(make_pdf("Hello   first page", "Second\npage here"))
    assert text == "Hello first page Second page here"


def test_extract_text_rejects_garbage():
    with pytest.raises(InvalidPDFError):
        extract_text(b"%PDF-1.4 definitely not a real pdf")


def test_extract_text_of_blank_pdf_is_empty():
    assert extract_text(make_pdf("")) == ""


def test_chunk_text_overlaps_windows():
    words = [f"w{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), size=200, overlap=40)
    assert len(chunks) == 2
    assert chunks[0].split() == words[:200]
    assert chunks[1].split() == words[160:]


def test_chunk_text_short_and_empty():
    assert chunk_text("just a few words") == ["just a few words"]
    assert chunk_text("") == []


def test_chunk_text_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)
