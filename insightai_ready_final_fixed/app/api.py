from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from app.pdf_utils import extract_clean_text_from_pdf
from app.vector_store import MockVectorStore
from app.ai_engine import OpenAIEngine
import shutil, uuid

router = APIRouter()
vector_store = MockVectorStore()
engine = OpenAIEngine(api_key="your-api-key")

@router.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    temp_filename = f"/tmp/{uuid.uuid4().hex}.pdf"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = extract_clean_text_from_pdf(temp_filename)
    vector_store.add_document(text)
    return {"message": "PDF uploaded and indexed", "tokens": len(text.split())}

class QueryRequest(BaseModel):
    query: str
    top_k: int = 1

@router.post("/query/")
async def query_pdf(data: QueryRequest):
    results = vector_store.search(data.query, k=data.top_k)
    context = results[0][0] if results else ""
    answer = engine.generate_answer(data.query, context)
    return {
        "query": data.query,
        "answer": answer,
        "results": [{"text": r[0], "score": r[1]} for r in results]
    }