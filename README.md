# InsightAI

[![CI](https://github.com/akincenk/insightai/actions/workflows/ci.yml/badge.svg)](https://github.com/akincenk/insightai/actions/workflows/ci.yml)

A small FastAPI service for asking questions about PDFs: upload documents, then get answers grounded in the passages most relevant to your question (retrieval-augmented generation).

## How it works

1. `POST /upload/` checks the file (`.pdf` extension, `%PDF-` header, size limit), extracts its text with PyMuPDF, and splits it into overlapping 200-word chunks.
   Documents that would produce more than 200 chunks are rejected before any OpenAI call is made.
2. Each chunk is embedded with the OpenAI Embeddings API (`text-embedding-3-small` by default) and stored in an in-memory NumPy matrix. The store holds at most 2000 chunks; when it is full, the oldest chunks are evicted first.
3. `POST /query/` embeds the question, ranks chunks by cosine similarity, and sends the top `k` chunks plus the question to the OpenAI Responses API. You get back the answer and the source chunks with their scores.

## Stack

- Python 3.11+, FastAPI, Uvicorn
- PyMuPDF for PDF text extraction
- NumPy for vector search (brute-force cosine similarity, no external vector DB)
- OpenAI Python SDK v1 (`OpenAI()` client, `embeddings.create`, `responses.create`)
- pytest + FastAPI `TestClient`, ruff, GitHub Actions

```
app/
  main.py         FastAPI app, CORS, error handlers, startup config check
  auth.py         X-API-Key check
  api.py          /upload/ and /query/ endpoints, upload validation
  pdf_utils.py    text extraction and chunking
  vector_store.py in-memory cosine-similarity store
  ai_engine.py    OpenAI embeddings and answer generation
tests/            unit and API tests (OpenAI is always mocked)
frontend/         optional example React component (see below)
```

## Quickstart

You need an OpenAI API key. Without one the server still starts, but `/upload/` and `/query/` return `503`.

You also need to set `API_KEY` (any long random string). The server refuses to start without it unless you set `ALLOW_NO_AUTH=true`, which is meant for local development only.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then set OPENAI_API_KEY and API_KEY in .env
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
- `API_KEY` (required): shared secret clients send in the `X-API-Key` header.
- `ALLOW_NO_AUTH` (default `false`): set to `true` to run without `API_KEY`. Local development only.
- `OPENAI_EMBEDDING_MODEL` (default `text-embedding-3-small`).
- `OPENAI_CHAT_MODEL` (default `gpt-5.4-mini`): any model that supports the Responses API.
- `MAX_UPLOAD_MB` (default `10`): maximum upload size.
- `MAX_CHUNKS_PER_UPLOAD` (default `200`): larger documents are rejected with `413`.
- `MAX_STORE_CHUNKS` (default `2000`): total chunks kept in memory; oldest are evicted first.
- `ALLOWED_ORIGINS`: comma-separated browser origins allowed by CORS. Empty means CORS is off.

## API

`/upload/` and `/query/` require the `X-API-Key` header and return `401` without a valid key. `GET /` does not.

`POST /upload/`: multipart form with a `file` field.

```bash
curl -H "X-API-Key: $API_KEY" -F file=@paper.pdf http://127.0.0.1:8000/upload/
# {"filename": "paper.pdf", "chunks": 12, "words": 2210}
```

Errors: `400` for a non-PDF or corrupt file, `413` if it is too large or has more than `MAX_CHUNKS_PER_UPLOAD` chunks, `422` if there is no extractable text (for example a scanned PDF), `503` if no API key is set, `502` if the OpenAI call fails.

`POST /query/`: JSON body with `query` (1-2000 chars) and `top_k` (1-10, default 3).

```bash
curl -X POST http://127.0.0.1:8000/query/ \
  -H "X-API-Key: $API_KEY" \
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

The tests never call OpenAI. They build real PDFs in memory with PyMuPDF and swap the embedding and answer functions for deterministic fakes. Any warning raised during the test run fails it (except one emitted by PyMuPDF's own bindings at import time). CI runs ruff, `pip-audit` on the runtime dependencies and pytest on Python 3.11 and 3.12, and builds the Docker image. Dependabot opens weekly update PRs for pip packages and GitHub Actions.

## Optional frontend

`frontend/InsightAIUI.jsx` is a single React component, not a standalone app, and it is not built or tested in CI. It expects a Vite project that already has React and [shadcn/ui](https://ui.shadcn.com/) (`@/components/ui/*`). It calls `VITE_API_URL` (default `http://localhost:8000`) and asks the user for the API key at runtime, so the key is never compiled into the bundle. To use it, add the backend origin to `ALLOWED_ORIGINS` (for example `http://localhost:5173`). The Swagger UI at `/docs` does the same job without any setup.

## Limitations and next steps

- The index lives in process memory. It is lost on restart and not shared between workers. A persistent vector store (pgvector, FAISS on disk, etc.) would fix that.
- Brute-force search is fine for a few thousand chunks. Beyond that, use an ANN index.
- Authentication is a single shared API key. There are no user accounts and no per-user isolation: everyone with the key shares one index.
- Eviction is first-in, first-out by chunk, so a document can be partially evicted when the store is full.
- Scanned (image-only) PDFs are rejected because there is no OCR.
- Chunking splits on word counts, not on sentence or section boundaries.

## Design decisions

- **No vector database.** A NumPy matrix and a dot product are enough for a capped store of 2000 chunks and keep the service to a single process with no extra infrastructure.
- **Caps instead of unbounded growth.** The per-upload and total chunk limits bound both memory use and OpenAI spend per request.
- **Sync handlers.** The endpoints are plain `def`, so FastAPI runs the blocking PDF parsing and OpenAI calls in its thread pool instead of on the event loop.
- **OpenAI behind one module.** `ai_engine.py` is the only code that talks to OpenAI, which keeps the tests free of network calls.

## Security

- `/upload/` and `/query/` spend OpenAI credits, so they require an `X-API-Key` header (compared in constant time). The app will not start without `API_KEY` unless `ALLOW_NO_AUTH=true` is set explicitly.
- Uploads are checked for extension, `%PDF-` magic bytes and size, and are parsed in memory; nothing is written to disk.
- CORS is off by default and, when enabled, allows only the listed origins and no credentials.
- Secrets come from the environment; `.env` is git-ignored. The Docker image runs as a non-root user.
- Uploaded text is sent to OpenAI. Do not upload documents you are not allowed to share with a third party.

## License

MIT
