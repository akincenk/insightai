# InsightAI

[![CI](https://github.com/akincenk/insightai/actions/workflows/ci.yml/badge.svg)](https://github.com/akincenk/insightai/actions/workflows/ci.yml)

A small FastAPI service for asking questions about PDFs: upload documents, then get answers grounded in the passages most relevant to your question (retrieval-augmented generation).

## How it works

1. `POST /upload/` checks the file (`.pdf` extension, `%PDF-` header, size limit), extracts its text with PyMuPDF, and splits it into overlapping 200-word chunks.
2. Each chunk is embedded with the OpenAI Embeddings API (`text-embedding-3-small` by default) and stored in an in-memory NumPy matrix.
3. `POST /query/` embeds the question, ranks chunks by cosine similarity, and sends the top `k` chunks plus the question to the OpenAI Responses API. You get back the answer and the source chunks with their scores.

## Stack

- Python 3.11+, FastAPI, Uvicorn
- PyMuPDF for PDF text extraction
- NumPy for vector search (brute-force cosine similarity, no external vector DB)
- OpenAI Python SDK v1 (`OpenAI()` client, `embeddings.create`, `responses.create`)
- pytest + FastAPI `TestClient`, ruff, GitHub Actions

```
app/
  main.py         FastAPI app, CORS, error handlers
  api.py          /upload/ and /query/ endpoints, upload validation
  pdf_utils.py    text extraction and chunking
  vector_store.py in-memory cosine-similarity store
  ai_engine.py    OpenAI embeddings and answer generation
tests/            unit and API tests (OpenAI is always mocked)
frontend/         optional example React component (see below)
```

## Quickstart

You need an OpenAI API key. Without one the server still starts, but `/upload/` and `/query/` return `503`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then set OPENAI_API_KEY in .env
set -a && source .env && set +a
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs to use the interactive Swagger UI.

### Docker

```bash
docker build -t insightai .
docker run --rm -p 8000:8000 --env-file .env insightai
```

The image runs as a non-root user.

## Configuration

Everything is configured through environment variables (see `.env.example`):

- `OPENAI_API_KEY` (required): read by the OpenAI SDK.
- `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`).
- `OPENAI_CHAT_MODEL` (default `gpt-5.4-mini`): any model that supports the Responses API.
- `MAX_UPLOAD_MB` (default `10`): maximum upload size.
- `ALLOWED_ORIGINS`: comma-separated browser origins allowed by CORS. Empty means CORS is off.

## API

`POST /upload/`: multipart form with a `file` field.

```bash
curl -F file=@paper.pdf http://127.0.0.1:8000/upload/
# {"filename": "paper.pdf", "chunks": 12, "words": 2210}
```

Errors: `400` for a non-PDF or corrupt file, `413` if it is too large, `422` if there is no extractable text (for example a scanned PDF), `503` if no API key is set, `502` if the OpenAI call fails.

`POST /query/`: JSON body with `query` (1-2000 chars) and `top_k` (1-10, default 3).

```bash
curl -X POST http://127.0.0.1:8000/query/ \
  -H 'Content-Type: application/json' \
  -d '{"query": "What dataset was used?", "top_k": 3}'
# {"query": "...", "answer": "...", "results": [{"text": "...", "source": "paper.pdf", "score": 0.83}, ...]}
```

Returns `400` if nothing has been uploaded yet.

`GET /`: health check.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check .
pytest
```

The tests never call OpenAI. They build real PDFs in memory with PyMuPDF and swap the embedding and answer functions for deterministic fakes. CI runs ruff and pytest on Python 3.11 and 3.12.

## Optional frontend

`frontend/InsightAIUI.jsx` is a single React component, not a standalone app. It expects a project that already has React and [shadcn/ui](https://ui.shadcn.com/) (`@/components/ui/*`), and it calls `http://localhost:8000`. To use it, add the backend origin to `ALLOWED_ORIGINS` (for example `http://localhost:5173`). The Swagger UI at `/docs` does the same job without any setup.

## Limitations and next steps

- The index lives in process memory. It is lost on restart and not shared between workers. A persistent vector store (pgvector, FAISS on disk, etc.) would fix that.
- Brute-force search is fine for a few thousand chunks. Beyond that, use an ANN index.
- There is no authentication or per-user isolation: everyone shares one index.
- Scanned (image-only) PDFs are rejected because there is no OCR.
- Chunking splits on word counts, not on sentence or section boundaries.

## License

MIT
