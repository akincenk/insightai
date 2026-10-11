import pymupdf


class InvalidPDFError(ValueError):
    pass


def extract_text(data: bytes) -> str:
    """Extract whitespace-normalised text from PDF bytes."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            text = " ".join(page.get_text() for page in doc)
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise InvalidPDFError(str(exc)) from exc
    return " ".join(text.split())


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping windows of `size` words."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = text.split()
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step) if words[i : i + size]]
