import pymupdf


class InvalidPDFError(ValueError):
    pass


def extract_pages(data: bytes) -> list[str]:
    """Whitespace-normalised text of every page, in order (page 1 is index 0)."""
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return [" ".join(page.get_text().split()) for page in doc]
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise InvalidPDFError(str(exc)) from exc


def extract_text(data: bytes) -> str:
    """Extract whitespace-normalised text from PDF bytes."""
    return " ".join(p for p in extract_pages(data) if p)


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into overlapping windows of `size` words."""
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")
    words = text.split()
    step = size - overlap
    return [" ".join(words[i : i + size]) for i in range(0, max(len(words) - overlap, 1), step) if words[i : i + size]]


def chunk_pages(pages: list[str], size: int = 200, overlap: int = 40) -> list[tuple[str, int]]:
    """Chunk page by page so every chunk keeps its 1-based page number for citations."""
    return [(c, n) for n, page in enumerate(pages, 1) for c in chunk_text(page, size, overlap)]
