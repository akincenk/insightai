# InsightAI

[![CI](https://github.com/akincenk/insightai/actions/workflows/ci.yml/badge.svg)](https://github.com/akincenk/insightai/actions/workflows/ci.yml)

A small FastAPI service for asking questions about PDFs: upload documents, then get answers grounded in the passages most relevant to your question (retrieval-augmented generation).

## How it works

1. `POST /upload/` checks the file (`.pdf` extension, `%PDF-` header, size limit), extracts the text of every page with PyMuPDF, and splits each page into overlapping 200-word chunks, so every chunk keeps its page number for citations.
   Documents that would produce more than 200 chunks are rejected before any OpenAI call is made.
2. Each chunk is embedded with the OpenAI Embeddings API (`text-embedding-3-small` by default) and stored in an in-memory NumPy matrix. The store holds at most 2000 chunks; when it is full, the oldest chunks are evicted first.
3. `POST /query/` retrieves the top `k` chunks in one of three modes (`mode` in the request body):
   - `vector` (default): cosine similarity over embeddings.
   - `keyword`: Okapi BM25 over content words (common words such as *a*, *can*, *the* are ignored; hyphenated codes such as `RX-17` stay one token).
   - `hybrid`: reciprocal rank fusion of the two.

   The chunks and the question go to the OpenAI Responses API, which must cite every claim as `[1]`, `[2]`. You get back the answer and numbered sources with file name, page and score, so each citation can be checked.

4. Optional: with `RERANK_URL` set, the 20 best candidates are reordered by a cross-encoder (a model that reads the question and each chunk together) before the top `k` go to the answer step. If the reranker fails, the retrieval order is kept and the response says `"reranked": false`.

The default mode was chosen by measurement, not by preference; see [Evaluation](#evaluation).

## Stack

- Python 3.11+, FastAPI, Uvicorn
- PyMuPDF for PDF text extraction
- NumPy for vector search (brute-force cosine similarity, no external vector DB) and a small pure-Python BM25 for keyword search
- OpenAI Python SDK v1 (`OpenAI()` client, `embeddings.create`, `responses.create`)
- pytest + FastAPI `TestClient`, ruff, GitHub Actions

```
app/
  main.py         FastAPI app, CORS, error handlers, startup config check
  auth.py         X-API-Key check
  api.py          /upload/ and /query/ endpoints, upload validation
  pdf_utils.py    text extraction and chunking
  vector_store.py in-memory store: vector, BM25 keyword and hybrid (RRF) search
  ai_engine.py    OpenAI embeddings and answer generation
eval/             retrieval evaluation: fictional handbook, 30 questions, recall@k and MRR
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
- `RERANK_URL` (optional): a `/v1/rerank` endpoint (llama.cpp `llama-server --reranking`, vLLM, Jina or a Cohere-compatible gateway). Read only from the environment, never from a request.
- `RERANK_MODEL`, `RERANK_API_KEY` (optional): sent to the reranker when set.
- `RERANK_CANDIDATES` (default `20`) and `RERANK_TIMEOUT` (default `30` seconds).

## API

`/upload/` and `/query/` require the `X-API-Key` header and return `401` without a valid key. `GET /` does not.

`POST /upload/`: multipart form with a `file` field.

```bash
curl -H "X-API-Key: $API_KEY" -F file=@paper.pdf http://127.0.0.1:8000/upload/
# {"filename": "paper.pdf", "chunks": 12, "words": 2210}
```

Errors: `400` for a non-PDF or corrupt file, `413` if it is too large or has more than `MAX_CHUNKS_PER_UPLOAD` chunks, `422` if there is no extractable text (for example a scanned PDF), `503` if no API key is set, `502` if the OpenAI call fails.

`POST /query/`: JSON body with `query` (1-2000 chars), `top_k` (1-10, default 3), `mode` (`vector`, `keyword` or `hybrid`, default `vector`) and `rerank` (default: on when `RERANK_URL` is set; `true` without a reranker returns `400`).

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
- Chunking splits on word counts, not on sentence or section boundaries, and a chunk does not carry its section's context; the reranker suffers from that most (see Evaluation).

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

## Evaluation

`eval/` holds a fictional employee handbook (10 documents, 25 pages) and 30 questions, each with the page that answers it. Some questions quote exact codes (`form RX-17`, `error E-42`); most are paraphrased ("My handheld cannot reach the stock system"). `python -m eval.run` builds the store, runs every question in every mode and reports recall@1/3/5 and MRR (mean reciprocal rank of the correct page).

Results with two local embedding models through Ollama (2026-11-25). EmbeddingGemma 2 ran as a local bf16 GGUF build (tag `embeddinggemma2-bf16`); all-MiniLM-L6 is the public `all-minilm` tag, so that row can be reproduced exactly:

| Embedding model | Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| EmbeddingGemma 2 | keyword | 0.67 | 0.90 | 0.90 | 0.78 |
| EmbeddingGemma 2 | **vector** | **0.90** | **1.00** | **1.00** | **0.94** |
| EmbeddingGemma 2 | hybrid | 0.83 | 0.93 | 0.97 | 0.90 |
| all-MiniLM-L6 | keyword | 0.67 | 0.90 | 0.90 | 0.78 |
| all-MiniLM-L6 | **vector** | **0.93** | **0.97** | **0.97** | **0.95** |
| all-MiniLM-L6 | hybrid | 0.90 | 0.93 | 0.93 | 0.92 |

What the numbers say:

- On this small handbook with clearly separated topics, dense retrieval alone is best with both models, so `vector` is the default.
- Keyword search finds exact codes but misses paraphrases; equal-weight fusion lets its weaker ranking pull good vector matches down.
- The first hybrid run exposed a real bug: function words counted as keyword hits and pushed the right page out of the top 10. Ignoring them raised keyword MRR from 0.74 to 0.78 and fixed that question.
- I did not tune weights or questions to make hybrid win; with 30 questions that would only fit the test set.

Run it yourself, fully local and free with [Ollama](https://ollama.com):

```bash
ollama pull all-minilm
OPENAI_BASE_URL=http://127.0.0.1:11434/v1 OPENAI_API_KEY=local \
OPENAI_EMBEDDING_MODEL=all-minilm python -m eval.run
```

`python -m eval.run --fake` runs offline with toy vectors; the test suite uses it to check the harness, not to score retrieval.

### Codes, reranking and the answers themselves (2026-11-26)

Three further questions, all measured locally with EmbeddingGemma 2, the `bge-reranker-v2-m3` cross-encoder served by `llama-server --reranking`, and Qwen 3.8 27B writing the answers through Ollama's Responses API.

**Do embeddings blur identifiers?** `python -m eval.run --set codes` uses a second fictional set built for exactly that case: a device manual of 24 near-identical pages that differ mainly in their error codes (`E-4127`, `E-4172`, `E-4217`...), with 24 questions that quote a code and 6 that only describe the symptom.

| Set | Mode | Recall@1 | Recall@3 | Recall@5 | MRR |
| --- | --- | --- | --- | --- | --- |
| Codes | keyword | 0.97 | 1.00 | 1.00 | 0.98 |
| Codes | vector | 0.97 | 0.97 | 1.00 | 0.97 |
| Codes | **hybrid** | **1.00** | **1.00** | **1.00** | **1.00** |
| Handbook | vector | 0.90 | 1.00 | 1.00 | 0.94 |
| Handbook | **vector + rerank** | **0.97** | 0.97 | 0.97 | **0.97** |
| Codes | vector + rerank | 1.00 | 1.00 | 1.00 | 1.00 |

- I expected dense retrieval to confuse codes that share their digits. It did not: EmbeddingGemma 2 found the right page first 29 times out of 30. Hybrid was the only mode that was perfect, so it is the better choice for identifier-heavy documents, but the gap is one question, not a different league.
- Reranking put the right page first more often (handbook recall@1 0.90 → 0.97) but pushed one page out of the top 10: "The fridge goods were too warm on arrival" is answered by a page that never says *fridge* or *warm* (it says "if a reading is outside the range", the range being on the page before). The cross-encoder reads each chunk alone, so it lost the context the embedding still had. Prefixing each chunk with its document name did not fix it and cost the codes set one question, so it is not used. That is why reranking is optional and off unless configured.

**Are the answers right, cited and honest?** `python -m eval.answers` asks the 30 handbook questions (each with the facts a correct answer must contain, in [`eval/answers.json`](eval/answers.json)) and 10 questions the handbook does not answer (parental leave, parking, salary day...).

| Pipeline | Correct | Cited | Citations pointing at the right page | Unanswerable: said "could not find" | False "could not find" |
| --- | --- | --- | --- | --- | --- |
| **vector, top 3** | **30 / 30** | 30 / 30 | 87% | **10 / 10** | **0 / 30** |
| vector + rerank, top 3 | 29 / 30 | 30 / 30 | 85% | 10 / 10 | 1 / 30 |

- Every answer was correct and cited, and the model never invented an answer to the 10 questions the documents do not cover; it said it could not find them and named what the excerpts did cover.
- 13% of citations point at a page other than the one that answers the question. I read all of them: they cite 5 distinct pages, 3 of them the previous page of the same document (for example the cold-chain rule before the excursion procedure) and 2 a related page of another document (protective equipment next to the forklift check, receiving counts next to stock counts). None is a made-up source, but none is the page that holds the answer either.
- Reranking cost one answer, the same fridge question: with the right page gone, the model correctly said it could not find the answer. A wrong retrieval became an honest "I don't know", not a wrong answer.
- Grading is lexical (every fact group must appear), so it is lenient where the fact is a plain yes or no; I read those answers by hand. Every answer is in [`eval/results/`](eval/results/).

