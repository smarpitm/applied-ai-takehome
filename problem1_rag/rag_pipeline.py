"""Core Problem 1 RAG pipeline: idempotent ingestion, filtered retrieval, refusal gate, cited QA."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from problem1_rag.chunker import DocumentChunk, chunk_document
from problem1_rag.embedder import LocalEmbedder
from problem1_rag.parsers import discover_corpus_files, parse_document
from problem1_rag.store_lancedb import LanceVectorStore, SearchHit
from problem1_rag.store_qdrant import QdrantVectorStore
from shared.config import ModelEndpointConfig, Settings, get_settings
from shared.llm_client import LLMRouter
from shared.logging_utils import append_jsonl, get_json_logger, log_json_event, utc_now_iso

NO_CONTEXT_MESSAGE = (
    "No relevant context was found in the indexed corpus above the similarity threshold "
    "to answer this question reliably."
)


@dataclass
class IngestSummary:
    """Result returned by corpus ingestion."""

    files_scanned: int
    files_inserted: int
    files_updated: int
    files_skipped_unchanged: int
    chunks_processed: int
    lancedb_vector_count: int
    qdrant_vector_count: int | None
    elapsed_ms: float


@dataclass
class RAGQueryResult:
    """Structured answer, citations, and telemetry for a single RAG query."""

    request_id: str
    query: str
    answer: str
    citations: list[str]
    valid_citations: list[str]
    no_context: bool
    returned_chunk_count: int
    hits: list[SearchHit]
    retrieval_latency_ms: float
    llm_latency_ms: float
    total_latency_ms: float
    input_tokens: int
    output_tokens: int
    provider_used: str
    model_used: str
    fallback_triggered: bool
    estimated_cost_usd: float
    store_used: str
    filters_applied: dict[str, str] | None


def extract_inline_citations(answer_text: str) -> list[str]:
    """Extract chunk IDs cited inline by the LLM (bracketed or bare chunk IDs)."""
    matches = re.findall(r"([A-Za-z0-9_.-]+__c\d{3}__[a-f0-9]{10})", answer_text)
    seen: list[str] = []
    for m in matches:
        if m not in seen:
            seen.append(m)
    return seen


class RAGService:
    """End-to-end RAG ingestion and query orchestrator."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: LocalEmbedder | None = None,
        lance_store: LanceVectorStore | None = None,
        qdrant_store: QdrantVectorStore | None = None,
        router: LLMRouter | None = None,
        llm_endpoint: ModelEndpointConfig | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embedder = embedder or LocalEmbedder.from_settings(self.settings)
        self.lance_store = lance_store or LanceVectorStore(
            db_path=self.settings.lancedb_path,
            table_name=self.settings.lancedb_table,
            embedding_dim=self.embedder.dim,
            embedding_model=self.embedder.model_name,
        )
        self.qdrant_store = qdrant_store
        self.router = router or LLMRouter(settings=self.settings)
        self.llm_endpoint = llm_endpoint or self.settings.p1_generator_config()
        self.logger = get_json_logger("applied_ai.rag", self.settings.log_level)
        self.query_log_path = self.settings.log_dir / "p1_queries.jsonl"

    def _get_qdrant(self) -> QdrantVectorStore:
        if self.qdrant_store is None:
            self.qdrant_store = QdrantVectorStore(
                collection_name=self.settings.qdrant_collection,
                embedding_dim=self.embedder.dim,
                embedding_model=self.embedder.model_name,
                url=self.settings.qdrant_url,
                local_path=self.settings.qdrant_path,
            )
        return self.qdrant_store

    def ingest_directory(
        self,
        corpus_dir: Path,
        chunk_size_tokens: int | None = None,
        chunk_overlap_tokens: int | None = None,
        sync_qdrant: bool = False,
    ) -> IngestSummary:
        """Idempotently parse, chunk, embed, and store all PDF/HTML/MD files under corpus_dir."""
        t0 = time.perf_counter()
        c_size = chunk_size_tokens or self.settings.chunk_size_tokens
        c_overlap = (
            self.settings.chunk_overlap_tokens
            if chunk_overlap_tokens is None
            else chunk_overlap_tokens
        )

        files = discover_corpus_files(Path(corpus_dir))
        inserted = 0
        updated = 0
        skipped = 0
        chunks_processed = 0

        existing_hashes = self.lance_store.get_source_hashes()
        qdrant = self._get_qdrant() if sync_qdrant else None

        for file_path in files:
            doc = parse_document(file_path, Path(corpus_dir))
            chunks: list[DocumentChunk] = chunk_document(
                doc, chunk_size_tokens=c_size, chunk_overlap_tokens=c_overlap
            )
            chunks_processed += len(chunks)

            prev = existing_hashes.get(doc.source_id)
            if prev and prev[0] == doc.doc_hash and prev[1] == len(chunks) and not sync_qdrant:
                skipped += 1
                continue

            vectors = self.embedder.embed_documents([c.text for c in chunks])
            status = self.lance_store.upsert_document(
                source_id=doc.source_id,
                doc_hash=doc.doc_hash,
                chunks=chunks,
                vectors=vectors,
            )
            if status == "inserted":
                inserted += 1
            elif status == "updated":
                updated += 1
            elif status == "skipped_unchanged":
                skipped += 1

            if qdrant is not None:
                qdrant.upsert_document(
                    source_id=doc.source_id,
                    doc_hash=doc.doc_hash,
                    chunks=chunks,
                    vectors=vectors,
                )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        summary = IngestSummary(
            files_scanned=len(files),
            files_inserted=inserted,
            files_updated=updated,
            files_skipped_unchanged=skipped,
            chunks_processed=chunks_processed,
            lancedb_vector_count=self.lance_store.count(),
            qdrant_vector_count=qdrant.count() if qdrant is not None else None,
            elapsed_ms=elapsed_ms,
        )
        log_json_event(self.logger, "rag_ingest", **asdict(summary))
        return summary

    def query(
        self,
        question: str,
        k: int | None = None,
        filters: dict[str, str] | None = None,
        store: str = "lancedb",
        min_similarity_score: float | None = None,
        log_event: bool = True,
    ) -> RAGQueryResult:
        """Run top-k filtered retrieval, enforce threshold refusal gate, and cite chunks inline."""
        req_id = uuid.uuid4().hex[:12]
        t_start = time.perf_counter()
        top_k = k or self.settings.top_k
        threshold = (
            self.settings.min_similarity_score
            if min_similarity_score is None
            else min_similarity_score
        )

        q_vec = self.embedder.embed_query(question)
        active_store = store.lower()
        if active_store == "qdrant":
            hits, retrieval_ms = self._get_qdrant().search(q_vec, k=top_k, filters=filters)
        elif active_store == "lancedb":
            hits, retrieval_ms = self.lance_store.search(q_vec, k=top_k, filters=filters)
        else:
            raise ValueError(f"Unsupported store '{store}'. Use 'lancedb' or 'qdrant'.")

        passing_hits = [h for h in hits if h.score >= threshold]

        if not passing_hits:
            total_ms = round((time.perf_counter() - t_start) * 1000, 2)
            result = RAGQueryResult(
                request_id=req_id,
                query=question,
                answer=NO_CONTEXT_MESSAGE,
                citations=[],
                valid_citations=[],
                no_context=True,
                returned_chunk_count=0,
                hits=hits,
                retrieval_latency_ms=retrieval_ms,
                llm_latency_ms=0.0,
                total_latency_ms=total_ms,
                input_tokens=0,
                output_tokens=0,
                provider_used="none_refused",
                model_used="none_refused",
                fallback_triggered=False,
                estimated_cost_usd=0.0,
                store_used=active_store,
                filters_applied=filters,
            )
            if log_event:
                self._emit_query_log(result)
            return result

        context_lines = [
            f"[{h.chunk_id}] (source={h.source_id}, section={h.section_header}, "
            f"score={h.score}):\n{h.text}"
            for h in passing_hits
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise engineering QA assistant. Answer ONLY using the retrieved "
                    "context chunks below. Always cite the exact chunk_id in square brackets like "
                    "[{chunk_id}] for every factual sentence (for example, [{example_id}]). "
                    "If the context does not contain the answer, state that it is not stated."
                ).replace("{example_id}", passing_hits[0].chunk_id),
            },
            {
                "role": "user",
                "content": (
                    "Retrieved Context:\n"
                    + "\n\n".join(context_lines)
                    + f"\n\nQuestion: {question}\n\n"
                    + "Answer concisely and include exact [chunk_id] citations:"
                ),
            },
        ]

        llm_resp = self.router.complete(self.llm_endpoint, messages, json_mode=False)
        cited_ids = extract_inline_citations(llm_resp.text)
        retrieved_ids = {h.chunk_id for h in passing_hits}
        valid_cites = [cid for cid in cited_ids if cid in retrieved_ids]
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)

        result = RAGQueryResult(
            request_id=req_id,
            query=question,
            answer=llm_resp.text.strip(),
            citations=cited_ids,
            valid_citations=valid_cites,
            no_context=False,
            returned_chunk_count=len(passing_hits),
            hits=passing_hits,
            retrieval_latency_ms=retrieval_ms,
            llm_latency_ms=llm_resp.latency_ms,
            total_latency_ms=total_ms,
            input_tokens=llm_resp.input_tokens,
            output_tokens=llm_resp.output_tokens,
            provider_used=llm_resp.provider_used,
            model_used=llm_resp.model_used,
            fallback_triggered=llm_resp.fallback_triggered,
            estimated_cost_usd=llm_resp.estimated_cost_usd,
            store_used=active_store,
            filters_applied=filters,
        )
        if log_event:
            self._emit_query_log(result)
        return result

    def _emit_query_log(self, res: RAGQueryResult) -> None:
        payload: dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "request_id": res.request_id,
            "store": res.store_used,
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
            "filters_applied": res.filters_applied,
            "citations": res.valid_citations,
        }
        log_json_event(self.logger, "rag_query", **payload)
        append_jsonl(self.query_log_path, payload)
