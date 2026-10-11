import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app import ai_engine
from app.auth import require_api_key
from app.pdf_utils import InvalidPDFError, chunk_text, extract_text
from app.vector_store import VectorStore

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
MAX_CHUNKS_PER_UPLOAD = int(os.getenv("MAX_CHUNKS_PER_UPLOAD", "200"))
MAX_STORE_CHUNKS = int(os.getenv("MAX_STORE_CHUNKS", "2000"))

router = APIRouter(dependencies=[Depends(require_api_key)])
store = VectorStore(max_chunks=MAX_STORE_CHUNKS)


@router.post("/upload/")
def upload_pdf(file: Annotated[UploadFile, File()]):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only .pdf files are accepted")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(400, "File is not a valid PDF")
    try:
        text = extract_text(data)
    except InvalidPDFError:
        raise HTTPException(400, "File is not a valid PDF") from None
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(422, "No extractable text found (scanned PDFs are not supported)")
    if len(chunks) > MAX_CHUNKS_PER_UPLOAD:
        # Checked before embedding so an oversized document never costs an OpenAI call.
        raise HTTPException(413, f"Document has {len(chunks)} chunks; the limit is {MAX_CHUNKS_PER_UPLOAD} per upload")
    store.add(chunks, ai_engine.embed(chunks), source=file.filename)
    return {"filename": file.filename, "chunks": len(chunks), "words": len(text.split())}


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/query/")
def query_pdf(data: QueryRequest):
    if not len(store):
        raise HTTPException(400, "No documents indexed yet; upload a PDF first")
    results = store.search(ai_engine.embed([data.query])[0], k=data.top_k)
    answer = ai_engine.answer(data.query, [text for text, _, _ in results])
    return {
        "query": data.query,
        "answer": answer,
        "results": [{"text": t, "source": s, "score": round(score, 4)} for t, s, score in results],
    }
