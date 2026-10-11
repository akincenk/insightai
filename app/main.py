from fastapi import FastAPI
from app.api import router

app = FastAPI()
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "InsightAI is running"}