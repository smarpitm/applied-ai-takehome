"""Bonus vector store: Self-hosted Qdrant with on-disk memmap + int8 scalar quantization."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from problem1_rag.chunker import DocumentChunk
from problem1_rag.store_lancedb import ALLOWED_FILTER_FIELDS, SearchHit


def _chunk_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    """Qdrant vector store supporting Docker server (memmap + int8) or local disk fallback."""

    def __init__(
        self,
        collection_name: str = "rag_chunks",
        embedding_dim: int = 384,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        url: str | None = "http://localhost:6333",
        local_path: Path = Path("./data/qdrant"),
    ) -> None:
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.mode_label = "local_disk_fallback"

        if url:
            try:
                client = QdrantClient(url=url, timeout=2.0)
                client.get_collections()
                self.client = client
                self.mode_label = "docker_memmap_int8"
            except Exception:
                Path(local_path).mkdir(parents=True, exist_ok=True)
                self.client = QdrantClient(path=str(local_path))
        else:
            Path(local_path).mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(local_path))

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection_name in existing:
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            ),
            hnsw_config=models.HnswConfigDiff(on_disk=True),
            optimizers_config=models.OptimizersConfigDiff(memmap_threshold=10_000),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=False,
                )
            ),
        )

    def count(self) -> int:
        """Return total points stored in the Qdrant collection."""
        res = self.client.count(collection_name=self.collection_name, exact=True)
        return int(res.count)

    def upsert_document(
        self,
        source_id: str,
        doc_hash: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> str:
        """Idempotently upsert document chunks into Qdrant."""
        if not chunks:
            return "skipped_empty"

        # Check existing points for this source_id
        existing, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id", match=models.MatchValue(value=source_id)
                    )
                ]
            ),
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        if existing:
            prev_hash = (existing[0].payload or {}).get("doc_hash")
            if prev_hash == doc_hash and len(existing) == len(chunks):
                return "skipped_unchanged"
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="source_id", match=models.MatchValue(value=source_id)
                            )
                        ]
                    )
                ),
            )
            status = "updated"
        else:
            status = "inserted"

        points = [
            models.PointStruct(
                id=_chunk_uuid(chunk.chunk_id),
                vector=[float(x) for x in vec],
                payload={
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "doc_hash": chunk.doc_hash,
                    "file_type": chunk.file_type,
                    "topic": chunk.topic,
                    "section_header": chunk.section_header,
                    "chunk_index": int(chunk.chunk_index),
                    "token_count": int(chunk.token_count),
                    "embedding_model": self.embedding_model,
                    "embedding_dim": int(self.embedding_dim),
                    "text": chunk.text,
                },
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)
        return status

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """Run Qdrant cosine search with optional metadata Filter."""
        q_filter = None
        if filters:
            conditions: list[models.Condition] = []
            for key, val in filters.items():
                if key not in ALLOWED_FILTER_FIELDS:
                    raise ValueError(f"Unsupported metadata filter '{key}'")
                conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=val))
                )
            q_filter = models.Filter(must=conditions)

        t0 = time.perf_counter()
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=q_filter,
            limit=k,
            with_payload=True,
        )
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        hits: list[SearchHit] = []
        for pt in res.points:
            p = pt.payload or {}
            hits.append(
                SearchHit(
                    chunk_id=str(p["chunk_id"]),
                    source_id=str(p["source_id"]),
                    file_type=str(p["file_type"]),
                    topic=str(p["topic"]),
                    section_header=str(p["section_header"]),
                    chunk_index=int(p["chunk_index"]),
                    token_count=int(p["token_count"]),
                    text=str(p["text"]),
                    score=round(max(0.0, min(1.0, float(pt.score))), 4),
                )
            )
        return hits, latency_ms

    def stats(self) -> dict[str, Any]:
        return {
            "store": "qdrant",
            "mode": self.mode_label,
            "collection_name": self.collection_name,
            "vector_count": self.count(),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }
