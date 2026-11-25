"""FastAPI service exposing Problem 1 ingestion, filtered RAG query, and stats endpoints."""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from problem1_rag.rag_pipeline import RAGService


class IngestRequest(BaseModel):
    corpus_dir: str = "./problem1_rag/data/corpus"
    chunk_size_tokens: int | None = Field(default=None, ge=64, le=4096)
    chunk_overlap_tokens: int | None = Field(default=None, ge=0, le=1024)
    sync_qdrant: bool = False


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, str] | None = None
    store: str = "lancedb"
    min_similarity_score: float | None = Field(default=None, ge=0.0, le=1.0)


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()


def create_app(service: RAGService | None = None) -> FastAPI:
    """Create FastAPI application, optionally injecting a custom RAGService for testing."""
    app = FastAPI(
        title="Cost-Efficient RAG Service (LanceDB + Qdrant)",
        version="0.1.0",
    )

    def _svc() -> RAGService:
        return service if service is not None else get_rag_service()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict[str, Any]:
        svc = _svc()
        data: dict[str, Any] = {"lancedb": svc.lance_store.stats()}
        if svc.qdrant_store is not None:
            data["qdrant"] = svc.qdrant_store.stats()
        return data

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict[str, Any]:
        svc = _svc()
        path = Path(req.corpus_dir)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Corpus directory not found: {path}")
        summary = svc.ingest_directory(
            corpus_dir=path,
            chunk_size_tokens=req.chunk_size_tokens,
            chunk_overlap_tokens=req.chunk_overlap_tokens,
            sync_qdrant=req.sync_qdrant,
        )
        return asdict(summary)

    @app.post("/query")
    def query(req: QueryRequest) -> dict[str, Any]:
        svc = _svc()
        try:
            res = svc.query(
                question=req.query,
                k=req.k,
                filters=req.filters,
                store=req.store,
                min_similarity_score=req.min_similarity_score,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "request_id": res.request_id,
            "query": res.query,
            "answer": res.answer,
            "citations": res.valid_citations,
            "all_raw_citations": res.citations,
            "no_context": res.no_context,
            "returned_chunk_count": res.returned_chunk_count,
            "retrieval_latency_ms": res.retrieval_latency_ms,
            "llm_latency_ms": res.llm_latency_ms,
            "total_latency_ms": res.total_latency_ms,
            "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens,
            "provider_used": res.provider_used,
            "model_used": res.model_used,
            "fallback_triggered": res.fallback_triggered,
            "estimated_cost_usd": res.estimated_cost_usd,
            "chunks": [asdict(h) for h in res.hits],
        }

    return app


app = create_app()
