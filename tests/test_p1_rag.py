"""Unit & integration tests for Problem 1 RAG pipeline, idempotency, IR metrics, and API."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from problem1_rag.api import create_app
from problem1_rag.chunker import chunk_document, make_chunk_id
from problem1_rag.cost_model import CostAssumptions, compute_cost_row
from problem1_rag.embedder import LocalEmbedder
from problem1_rag.eval_harness import compute_em_f1, compute_ir_metrics
from problem1_rag.parsers import ParsedDocument, ParsedSection, parse_document
from problem1_rag.rag_pipeline import RAGService
from problem1_rag.store_lancedb import LanceVectorStore
from shared.config import Settings
from shared.llm_client import LLMRouter


def _keyword_embed(texts: Sequence[str]) -> list[list[float]]:
    """Deterministic lightweight 384-dim hashing embedder for fast offline tests."""
    vecs: list[list[float]] = []
    for text in texts:
        vec = [0.0] * 384
        for tok in text.lower().split():
            idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % 384
            vec[idx] += 1.0
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        vecs.append([x / norm for x in vec])
    return vecs


def test_deterministic_chunk_id_and_overlap() -> None:
    long_text = " ".join(f"token{i}" for i in range(650))
    doc = ParsedDocument(
        source_id="architecture/test_doc.md",
        file_type="md",
        title="Test",
        topic="architecture",
        doc_hash="abc",
        sections=[ParsedSection(section_header="Overview", text=long_text)],
    )
    chunks = chunk_document(doc, chunk_size_tokens=200, chunk_overlap_tokens=32)
    assert len(chunks) >= 3
    assert chunks[0].chunk_id == make_chunk_id("architecture/test_doc.md", 0)
    assert chunks[0].token_count <= 200
    assert chunks[1].chunk_id == make_chunk_id("architecture/test_doc.md", 1)


def test_idempotent_reingest_and_stale_cleanup(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "security").mkdir(parents=True)
    md_path = corpus / "security" / "auth.md"
    md_path.write_text("# Auth\n\n## mTLS\n\nCertificates expire in 24 hours.", encoding="utf-8")

    html_path = corpus / "security" / "kms.html"
    html_path.write_text(
        "<html><head><title>KMS</title></head><body><nav>Nav</nav>"
        "<main><h2>Keys</h2><p>AES-256-GCM keys rotate yearly.</p></main></body></html>",
        encoding="utf-8",
    )

    embedder = LocalEmbedder(dim=384, embed_fn=_keyword_embed)
    settings = Settings(
        lancedb_path=tmp_path / "lancedb",
        log_dir=tmp_path / "logs",
        groq_api_key="dummy",
        chunk_size_tokens=128,
        chunk_overlap_tokens=16,
    )
    store = LanceVectorStore(db_path=settings.lancedb_path, embedding_dim=384)
    svc = RAGService(settings=settings, embedder=embedder, lance_store=store)

    first = svc.ingest_directory(corpus)
    assert first.files_scanned == 2
    assert first.files_inserted == 2
    count_after_first = store.count()

    # Re-ingest exact same corpus -> zero duplicates, both skipped
    second = svc.ingest_directory(corpus)
    assert second.files_skipped_unchanged == 2
    assert store.count() == count_after_first

    # Modify file to be smaller -> stale chunks deleted and replaced cleanly
    md_path.write_text("# Auth\n\n## mTLS\n\nRotated at 12 hours.", encoding="utf-8")
    third = svc.ingest_directory(corpus)
    assert third.files_updated == 1
    assert third.files_skipped_unchanged == 1
    assert store.count() == count_after_first

    parsed_html = parse_document(html_path, corpus)
    assert "Nav" not in parsed_html.sections[0].text


def test_no_context_refusal_blocks_llm_and_filters_work(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    (corpus / "ops").mkdir(parents=True)
    (corpus / "ops" / "slo.md").write_text(
        "# SLO\n\n## Targets\n\nQuery API availability SLO is 99.95% monthly.",
        encoding="utf-8",
    )

    llm_calls = 0

    def fake_transport(
        provider: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int, int]:
        nonlocal llm_calls
        llm_calls += 1
        cid = make_chunk_id("ops/slo.md", 0)
        return f"The availability SLO is 99.95% [{cid}].", 100, 15

    embedder = LocalEmbedder(dim=384, embed_fn=_keyword_embed)
    settings = Settings(
        lancedb_path=tmp_path / "lancedb",
        log_dir=tmp_path / "logs",
        groq_api_key="dummy",
        min_similarity_score=0.20,
    )
    router = LLMRouter(settings=settings, transport=fake_transport)
    store = LanceVectorStore(db_path=settings.lancedb_path, embedding_dim=384)
    svc = RAGService(settings=settings, embedder=embedder, lance_store=store, router=router)
    svc.ingest_directory(corpus)

    # Answerable with matching metadata filter
    res = svc.query("What is the Query API availability SLO?", k=3, filters={"topic": "ops"})
    assert not res.no_context
    assert llm_calls == 1
    assert res.valid_citations == [make_chunk_id("ops/slo.md", 0)]

    # Out-of-corpus query with threshold short-circuit -> 0 additional LLM calls
    refused = svc.query("What is on today's cafeteria menu?", k=3, min_similarity_score=0.90)
    assert refused.no_context is True
    assert llm_calls == 1
    assert refused.input_tokens == 0

    # FastAPI TestClient smoke test
    app = create_app(service=svc)
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/stats").json()["lancedb"]["vector_count"] == 1
    q_resp = client.post("/query", json={"query": "availability SLO", "k": 2})
    assert q_resp.status_code == 200
    assert q_resp.json()["returned_chunk_count"] >= 1


def test_ir_metrics_em_f1_and_cost_model() -> None:
    ir = compute_ir_metrics(["c2", "c1", "c3"], {"c1", "c4"}, k=3)
    assert ir["recall_at_k"] == 0.5
    assert ir["hit_rate"] == 1.0
    assert ir["mrr"] == 0.5
    assert ir["context_precision"] == round(1 / 3, 4)

    em, f1 = compute_em_f1("48 bytes [doc__c000__abcdef12]", "48 bytes")
    assert em == 1.0 and f1 == 1.0

    row_10m = compute_cost_row(10_000_000, "10M", CostAssumptions())
    assert row_10m.raw_1536d_vectors_gb == round((10_000_000 * 1536 * 4) / (1024**3), 3)
    assert row_10m.lancedb_total_usd < row_10m.pinecone_pod_usd
    assert row_10m.savings_vs_pinecone_pod_pct > 90.0


def test_qdrant_bonus_store_idempotency_and_filter(tmp_path: Path) -> None:
    from problem1_rag.store_qdrant import QdrantVectorStore

    doc = ParsedDocument(
        source_id="security/qdrant_doc.md",
        file_type="md",
        title="Qdrant",
        topic="security",
        doc_hash="q1",
        sections=[ParsedSection(section_header="Memmap", text="Qdrant uses int8 scalar quant.")],
    )
    chunks = chunk_document(doc, chunk_size_tokens=128, chunk_overlap_tokens=16)
    vecs = _keyword_embed([c.text for c in chunks])
    q_store = QdrantVectorStore(
        collection_name="test_q",
        embedding_dim=384,
        url=None,
        local_path=tmp_path / "qdrant",
    )
    assert q_store.upsert_document(doc.source_id, doc.doc_hash, chunks, vecs) == "inserted"
    assert q_store.upsert_document(doc.source_id, doc.doc_hash, chunks, vecs) == "skipped_unchanged"
    assert q_store.count() == len(chunks)
    hits, _ = q_store.search(vecs[0], k=2, filters={"topic": "security"})
    assert len(hits) == 1
    assert hits[0].chunk_id == chunks[0].chunk_id
