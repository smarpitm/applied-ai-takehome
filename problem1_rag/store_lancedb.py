"""Primary vector store: embedded LanceDB with SQL pre-filtering and idempotent merge_insert."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

from problem1_rag.chunker import DocumentChunk


@dataclass
class SearchHit:
    """Retrieved chunk paired with similarity score and metadata."""

    chunk_id: str
    source_id: str
    file_type: str
    topic: str
    section_header: str
    chunk_index: int
    token_count: int
    text: str
    score: float


ALLOWED_FILTER_FIELDS = {"source_id", "file_type", "topic", "section_header"}


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def build_sql_filter(filters: dict[str, str] | None) -> str | None:
    """Build safe SQL WHERE predicate for LanceDB metadata filtering."""
    if not filters:
        return None
    clauses: list[str] = []
    for key, val in filters.items():
        if key not in ALLOWED_FILTER_FIELDS:
            raise ValueError(
                f"Unsupported metadata filter '{key}'. Allowed: {sorted(ALLOWED_FILTER_FIELDS)}"
            )
        if not re.match(r"^[A-Za-z0-9_./:\- ]+$", val):
            raise ValueError(f"Invalid characters in filter value for '{key}'")
        clauses.append(f"{key} = '{_escape_sql_literal(val)}'")
    return " AND ".join(clauses) if clauses else None


class LanceVectorStore:
    """Disk-native columnar LanceDB store supporting idempotent re-ingestion."""

    def __init__(
        self,
        db_path: Path,
        table_name: str = "rag_chunks",
        embedding_dim: int = 384,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.table_name = table_name
        self.embedding_dim = embedding_dim
        self.embedding_model = embedding_model
        self.db = lancedb.connect(str(self.db_path))
        self.schema = pa.schema(
            [
                pa.field("chunk_id", pa.string()),
                pa.field("source_id", pa.string()),
                pa.field("doc_hash", pa.string()),
                pa.field("file_type", pa.string()),
                pa.field("topic", pa.string()),
                pa.field("section_header", pa.string()),
                pa.field("chunk_index", pa.int32()),
                pa.field("token_count", pa.int32()),
                pa.field("embedding_model", pa.string()),
                pa.field("embedding_dim", pa.int32()),
                pa.field("text", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.embedding_dim)),
            ]
        )

    def _table_exists(self) -> bool:
        return self.table_name in self.db.list_tables().tables

    def _get_or_create_table(self) -> lancedb.table.Table:
        if self._table_exists():
            return self.db.open_table(self.table_name)
        return self.db.create_table(self.table_name, schema=self.schema)

    def count(self) -> int:
        """Return total stored vector rows."""
        if not self._table_exists():
            return 0
        return int(self.db.open_table(self.table_name).count_rows())

    def get_source_hashes(self) -> dict[str, tuple[str, int]]:
        """Return mapping of source_id -> (doc_hash, row_count) currently stored."""
        if not self._table_exists():
            return {}
        tbl = self.db.open_table(self.table_name)
        if tbl.count_rows() == 0:
            return {}
        arrow_tbl = tbl.to_arrow().select(["source_id", "doc_hash"])
        hashes: dict[str, tuple[str, int]] = {}
        sources = arrow_tbl["source_id"].to_pylist()
        doc_hashes = arrow_tbl["doc_hash"].to_pylist()
        for src, h in zip(sources, doc_hashes, strict=True):
            prev = hashes.get(src)
            hashes[src] = (h, (prev[1] + 1) if prev else 1)
        return hashes

    def upsert_document(
        self,
        source_id: str,
        doc_hash: str,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> str:
        """Idempotently store chunks for a source_id.

        Returns one of: 'skipped_unchanged', 'updated', 'inserted'.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have identical length")
        if not chunks:
            return "skipped_empty"

        tbl = self._get_or_create_table()
        existing = self.get_source_hashes().get(source_id)
        if existing and existing[0] == doc_hash and existing[1] == len(chunks):
            return "skipped_unchanged"

        # If document changed (or chunk count changed), remove old rows for source_id first
        # so shrinking documents never retain orphan tail chunks.
        status = "inserted"
        if existing is not None:
            tbl.delete(f"source_id = '{_escape_sql_literal(source_id)}'")
            status = "updated"

        rows: list[dict[str, Any]] = []
        for chunk, vec in zip(chunks, vectors, strict=True):
            rows.append(
                {
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
                    "vector": [float(x) for x in vec],
                }
            )

        (
            tbl.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        return status

    def ensure_ivf_pq_index(
        self,
        num_partitions: int = 16,
        num_sub_vectors: int = 48,
    ) -> bool:
        """Create IVF-PQ index when dataset is large enough (>=256 rows for 8-bit PQ)."""
        if not self._table_exists():
            return False
        tbl = self.db.open_table(self.table_name)
        n_rows = tbl.count_rows()
        if n_rows < 256:
            return False
        parts = min(num_partitions, max(2, int(n_rows**0.5)))
        tbl.create_index(
            metric="cosine",
            index_type="IVF_PQ",
            num_partitions=parts,
            num_sub_vectors=num_sub_vectors,
            replace=True,
        )
        return True

    def search(
        self,
        query_vector: list[float],
        k: int = 5,
        filters: dict[str, str] | None = None,
    ) -> tuple[list[SearchHit], float]:
        """Run cosine similarity search with optional SQL pre-filtering.

        Returns (hits, retrieval_latency_ms).
        """
        t0 = time.perf_counter()
        if not self._table_exists():
            return [], 0.0
        tbl = self.db.open_table(self.table_name)
        if tbl.count_rows() == 0:
            return [], round((time.perf_counter() - t0) * 1000, 2)

        query_builder = tbl.search(query_vector).metric("cosine")
        sql_where = build_sql_filter(filters)
        if sql_where:
            query_builder = query_builder.where(sql_where, prefilter=True)
        rows = query_builder.limit(k).to_list()
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        hits: list[SearchHit] = []
        for row in rows:
            distance = float(row.get("_distance", 1.0))
            similarity = max(0.0, min(1.0, 1.0 - distance))
            hits.append(
                SearchHit(
                    chunk_id=str(row["chunk_id"]),
                    source_id=str(row["source_id"]),
                    file_type=str(row["file_type"]),
                    topic=str(row["topic"]),
                    section_header=str(row["section_header"]),
                    chunk_index=int(row["chunk_index"]),
                    token_count=int(row["token_count"]),
                    text=str(row["text"]),
                    score=round(similarity, 4),
                )
            )
        return hits, latency_ms

    def stats(self) -> dict[str, Any]:
        """Return index statistics and metadata breakdown."""
        hashes = self.get_source_hashes()
        return {
            "store": "lancedb",
            "db_path": str(self.db_path),
            "table_name": self.table_name,
            "vector_count": self.count(),
            "document_count": len(hashes),
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
        }
