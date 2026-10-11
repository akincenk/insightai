import os

import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.ai_engine import MissingAPIKeyError
from app.api import router

app = FastAPI(title="InsightAI", description="Ask questions about uploaded PDFs.")
app.include_router(router)

origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.exception_handler(MissingAPIKeyError)
def missing_key(_, exc):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(openai.OpenAIError)
def openai_error(_, exc):
    return JSONResponse(status_code=502, content={"detail": "Upstream OpenAI request failed"})


@app.get("/")
def root():
    return {"message": "InsightAI is running"}
