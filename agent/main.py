"""
main.py
FastAPI entrypoint. Run with:
    uvicorn agent.main:app --reload
Then open http://127.0.0.1:8000/docs to test it interactively - no frontend needed yet.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from backend.app.schemas.schemas import ImageObject, QueryResponse
from agent.controller import run_query

app = FastAPI(title="SatQuery AI - Agent Service")


class QueryRequest(BaseModel):
    query: str
    images: List[ImageObject]


@app.get("/")
def health_check():
    return {"status": "ok", "service": "SatQuery AI agent"}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    return run_query(query=request.query, images=request.images)
