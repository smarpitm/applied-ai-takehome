"""Three-layer evaluation harness for Problem 1: Retrieval IR, Answer quality, Latency & Cost."""

from __future__ import annotations

import json
import math
import random
import re
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from problem1_rag.chunker import DocumentChunk, chunk_document
from problem1_rag.cost_model import write_cost_artifacts
from problem1_rag.embedder import LocalEmbedder
from problem1_rag.parsers import discover_corpus_files, parse_document
from problem1_rag.rag_pipeline import RAGService
from problem1_rag.store_lancedb import LanceVectorStore, SearchHit
from problem1_rag.store_qdrant import QdrantVectorStore
from shared.config import Settings, get_settings
from shared.llm_client import LLMRouter


class RagAnswerJudgeVerdict(BaseModel):
    """Strict JSON schema for Problem 1 answer quality LLM judge."""

    evidence_quotes: list[str] = Field(default_factory=list)
    faithfulness_rationale: str = Field(..., min_length=1)
    faithfulness_score: int = Field(..., ge=1, le=5)
    relevance_rationale: str = Field(..., min_length=1)
    relevance_score: int = Field(..., ge=1, le=5)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"\[[A-Za-z0-9_.-]+__c\d{3}__[a-f0-9]+\]", " ", text.lower())
    cleaned = re.sub(r"[^a-z0-9\s.]", " ", cleaned)
    return " ".join(cleaned.split())


def compute_em_f1(prediction: str, gold: str) -> tuple[float, float]:
    """Compute normalized Exact Match (0/1) and token-level F1 against a gold answer."""
    pred_norm = _normalize_text(prediction)
    gold_norm = _normalize_text(gold)
    if not gold_norm:
        return (1.0 if not pred_norm else 0.0), (1.0 if not pred_norm else 0.0)
    em = 1.0 if pred_norm == gold_norm else 0.0
    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()
    if not pred_tokens or not gold_tokens:
        return em, 0.0
    common: dict[str, int] = {}
    for tok in pred_tokens:
        if tok in gold_tokens:
            common[tok] = min(pred_tokens.count(tok), gold_tokens.count(tok))
    overlap = sum(common.values())
    if overlap == 0:
        return em, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return em, round(f1, 4)


def compute_ir_metrics(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> dict[str, float]:
    """Compute Recall@k, Hit Rate@k, MRR, nDCG@k, and Context Precision@k."""
    if not relevant_ids:
        return {
            "recall_at_k": 0.0,
            "hit_rate": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
            "context_precision": 0.0,
        }
    top = retrieved_ids[:k]
    hits = [1 if cid in relevant_ids else 0 for cid in top]

    recall = sum(hits) / len(relevant_ids)
    hit_rate = 1.0 if any(hits) else 0.0

    mrr = 0.0
    for rank, h in enumerate(hits, start=1):
        if h == 1:
            mrr = 1.0 / rank
            break

    dcg = sum(h / math.log2(rank + 1) for rank, h in enumerate(hits, start=1))
    ideal_len = min(k, len(relevant_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_len + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    # Context precision: fraction of retrieved chunks up to k that are relevant (precision@k)
    context_precision = (sum(hits) / len(top)) if top else 0.0
    return {
        "recall_at_k": round(recall, 4),
        "hit_rate": round(hit_rate, 4),
        "mrr": round(mrr, 4),
        "ndcg_at_k": round(ndcg, 4),
        "context_precision": round(context_precision, 4),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return round(ordered[idx], 2)


def _judge_rag_answer(
    router: LLMRouter,
    settings: Settings,
    question: str,
    answer: str,
    hits: list[SearchHit],
) -> tuple[RagAnswerJudgeVerdict | None, str]:
    """Use separate P1 judge model with strict JSON output for faithfulness + relevance."""
    ctx_text = "\n\n".join(f"[{h.chunk_id}] {h.text}" for h in hits)
    schema = json.dumps(
        {
            "evidence_quotes": ["<quote from context>"],
            "faithfulness_rationale": "<verify every claim against context>",
            "faithfulness_score": "<int 1-5>",
            "relevance_rationale": "<verify whether answer directly addresses question>",
            "relevance_score": "<int 1-5>",
        },
        indent=2,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict RAG answer quality judge. Score Faithfulness (1-5: 5=every claim "
                "supported by retrieved context; 1=hallucinated facts) and Answer Relevance (1-5: "
                "5=directly and completely answers question; 1=off-topic). Output ONLY valid JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Retrieved Context:\n{ctx_text}\n\n"
                f"Candidate Answer:\n{answer}\n\n"
                f"Required JSON Schema:\n{schema}"
            ),
        },
    ]
    resp = router.complete(settings.p1_judge_config(), messages, json_mode=True, temperature=0.0)
    raw = resp.text.strip()
    try:
        return RagAnswerJudgeVerdict.model_validate_json(raw), resp.model_used
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            try:
                return RagAnswerJudgeVerdict.model_validate_json(match.group(0)), resp.model_used
            except Exception:
                pass
    return None, resp.model_used


def run_chunking_ablation(
    corpus_dir: Path,
    questions: list[dict[str, Any]],
    embedder: LocalEmbedder,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Compare chunk sizes/overlaps (256/32 vs 512/64 vs 1024/128) on retrieval IR metrics."""
    configs = [(256, 32), (512, 64), (1024, 128)]
    answerable = [q for q in questions if not q.get("unanswerable", False)]
    files = discover_corpus_files(corpus_dir)
    docs = [parse_document(f, corpus_dir) for f in files]

    results: list[dict[str, Any]] = []
    for c_size, c_overlap in configs:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            store = LanceVectorStore(
                db_path=Path(tmp) / "lance",
                embedding_dim=embedder.dim,
                embedding_model=embedder.model_name,
            )
            total_chunks = 0
            all_chunks: list[DocumentChunk] = []
            for doc in docs:
                chunks = chunk_document(
                    doc, chunk_size_tokens=c_size, chunk_overlap_tokens=c_overlap
                )
                vecs = embedder.embed_documents([c.text for c in chunks])
                store.upsert_document(doc.source_id, doc.doc_hash, chunks, vecs)
                total_chunks += len(chunks)
                all_chunks.extend(chunks)

            per_q: list[dict[str, float]] = []
            for q in answerable:
                q_vec = embedder.embed_query(q["question"])
                hits, _ = store.search(q_vec, k=k, filters=q.get("filters"))
                # Map ground-truth relevance across chunk boundaries via source_id + answer_keywords
                rel_sources = set(q.get("relevant_sources", []))
                keywords = [kw.lower() for kw in q.get("answer_keywords", [])]
                rel_ids = {
                    c.chunk_id
                    for c in all_chunks
                    if c.source_id in rel_sources
                    and (not keywords or any(kw in c.text.lower() for kw in keywords))
                }
                if not rel_ids:
                    rel_ids = {c.chunk_id for c in all_chunks if c.source_id in rel_sources}
                per_q.append(compute_ir_metrics([h.chunk_id for h in hits], rel_ids, k=k))

            results.append(
                {
                    "chunk_size_tokens": c_size,
                    "chunk_overlap_tokens": c_overlap,
                    "total_chunks": total_chunks,
                    "recall_at_k": round(statistics.mean(r["recall_at_k"] for r in per_q), 4),
                    "mrr": round(statistics.mean(r["mrr"] for r in per_q), 4),
                    "ndcg_at_k": round(statistics.mean(r["ndcg_at_k"] for r in per_q), 4),
                    "context_precision": round(
                        statistics.mean(r["context_precision"] for r in per_q), 4
                    ),
                }
            )
    return results


def run_scale_latency_benchmark(
    embedder: LocalEmbedder,
    num_vectors: int = 10_000,
    num_queries: int = 30,
) -> dict[str, Any]:
    """Benchmark warm/cold p50 and p95 latency on 10K synthetic 384-dim vectors."""
    rng = random.Random(42)
    dim = embedder.dim

    def _rand_vec() -> list[float]:
        raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    queries = [_rand_vec() for _ in range(num_queries)]
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        lance = LanceVectorStore(db_path=tmp_path / "lance_10k", embedding_dim=dim)
        tbl = lance._get_or_create_table()
        batch_rows = []
        for i in range(num_vectors):
            batch_rows.append(
                {
                    "chunk_id": f"syn_{i:06d}",
                    "source_id": f"synthetic/doc_{i // 10:04d}.md",
                    "doc_hash": "syn",
                    "file_type": "md",
                    "topic": ["architecture", "security", "operations", "api_reference"][i % 4],
                    "section_header": "synthetic",
                    "chunk_index": i % 10,
                    "token_count": 128,
                    "embedding_model": embedder.model_name,
                    "embedding_dim": dim,
                    "text": f"Synthetic chunk payload {i}",
                    "vector": _rand_vec(),
                }
            )
        tbl.add(batch_rows)

        # Flat LanceDB scan latencies
        flat_lat: list[float] = []
        for qv in queries:
            _, ms = lance.search(qv, k=5, filters={"topic": "security"})
            flat_lat.append(ms)

        # Build IVF-PQ index (256-code PQ compression)
        t_idx = time.perf_counter()
        lance.ensure_ivf_pq_index(num_partitions=64, num_sub_vectors=48)
        index_build_s = round(time.perf_counter() - t_idx, 2)

        # Re-open table cleanly to capture cold first-read + warm latencies
        cold_lance = LanceVectorStore(db_path=tmp_path / "lance_10k", embedding_dim=dim)
        ivfpq_lat: list[float] = []
        for qv in queries:
            _, ms = cold_lance.search(qv, k=5, filters={"topic": "security"})
            ivfpq_lat.append(ms)

        # Measure Qdrant local/Docker latency on 2,000-vector subset for fast CI runtime
        q_lat: list[float] = []
        qdrant = QdrantVectorStore(
            collection_name="bench_scale",
            embedding_dim=dim,
            url=None,
            local_path=tmp_path / "qdrant_bench",
        )
        q_chunks = [
            DocumentChunk(
                chunk_id=f"syn_{i:06d}",
                source_id="syn.md",
                doc_hash="syn",
                file_type="md",
                topic=["architecture", "security"][i % 2],
                section_header="synthetic",
                chunk_index=i,
                token_count=128,
                text=f"Synthetic chunk {i}",
            )
            for i in range(1000)
        ]
        qdrant.upsert_document("syn.md", "syn", q_chunks, [r["vector"] for r in batch_rows[:1000]])
        for qv in queries[:20]:
            _, ms = qdrant.search(qv, k=5, filters={"topic": "security"})
            q_lat.append(ms)
        q_mode = qdrant.mode_label
        qdrant.close()

        return {
            "synthetic_vector_count": num_vectors,
            "queries_tested": num_queries,
            "lancedb_flat_p50_ms": _percentile(flat_lat, 50),
            "lancedb_flat_p95_ms": _percentile(flat_lat, 95),
            "lancedb_ivfpq_index_build_s": index_build_s,
            "lancedb_ivfpq_cold_first_ms": ivfpq_lat[0] if ivfpq_lat else 0.0,
            "lancedb_ivfpq_p50_ms": _percentile(ivfpq_lat, 50),
            "lancedb_ivfpq_p95_ms": _percentile(ivfpq_lat, 95),
            "qdrant_1k_mode": q_mode,
            "qdrant_1k_p50_ms": _percentile(q_lat, 50),
            "qdrant_1k_p95_ms": _percentile(q_lat, 95),
        }


def run_full_evaluation(
    corpus_dir: Path,
    questions_path: Path,
    out_dir: Path,
    k: int = 5,
    settings: Settings | None = None,
    service: RAGService | None = None,
    sync_qdrant: bool = True,
    run_scale_bench: bool = True,
) -> dict[str, Any]:
    """Run complete 3-layer RAG eval, idempotency proof, Qdrant comparison, and cost export."""
    cfg = settings or get_settings()
    svc = service or RAGService(settings=cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    questions: list[dict[str, Any]] = json.loads(questions_path.read_text(encoding="utf-8"))

    # 1. Ingest + Idempotent re-ingest proof
    first_ingest = svc.ingest_directory(corpus_dir, sync_qdrant=sync_qdrant)
    second_ingest = svc.ingest_directory(corpus_dir, sync_qdrant=sync_qdrant)

    # 2a. Run all 24 RAG queries first (batches generator model loading on local GPU)
    per_question_rows: list[dict[str, Any]] = []
    ir_list: list[dict[str, float]] = []
    faith_scores: list[int] = []
    rel_scores: list[int] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    retrieval_latencies: list[float] = []
    e2e_latencies: list[float] = []
    qdrant_latencies: list[float] = []
    qdrant_ir_list: list[dict[str, float]] = []
    unanswerable_total = 0
    unanswerable_refused = 0
    judge_model_used = "none"

    query_records: list[tuple[dict[str, Any], Any]] = []
    for q in questions:
        is_unans = bool(q.get("unanswerable", False))
        q_filters = q.get("filters")
        res = svc.query(q["question"], k=k, filters=q_filters, store="lancedb")
        retrieval_latencies.append(res.retrieval_latency_ms)
        e2e_latencies.append(res.total_latency_ms)

        if sync_qdrant and not is_unans:
            q_hits, q_ms = svc._get_qdrant().search(
                svc.embedder.embed_query(q["question"]), k=k, filters=q_filters
            )
            qdrant_latencies.append(q_ms)
            qdrant_ir_list.append(
                compute_ir_metrics(
                    [h.chunk_id for h in q_hits],
                    set(q.get("relevant_chunk_ids", [])),
                    k=k,
                )
            )
        query_records.append((q, res))

    # 2b. Score all answerable outputs with separate strict-JSON judge model
    for q, res in query_records:
        is_unans = bool(q.get("unanswerable", False))
        q_filters = q.get("filters")
        if is_unans:
            unanswerable_total += 1
            if res.no_context:
                unanswerable_refused += 1
            per_question_rows.append(
                {
                    "id": q["id"],
                    "unanswerable": True,
                    "no_context_triggered": res.no_context,
                    "retrieval_latency_ms": res.retrieval_latency_ms,
                    "total_latency_ms": res.total_latency_ms,
                    "top_score": res.hits[0].score if res.hits else 0.0,
                }
            )
            continue

        rel_ids = set(q.get("relevant_chunk_ids", []))
        ir = compute_ir_metrics([h.chunk_id for h in res.hits], rel_ids, k=k)
        ir_list.append(ir)

        gold = q.get("gold_answer", "")
        em, f1 = compute_em_f1(res.answer, gold)
        em_scores.append(em)
        f1_scores.append(f1)

        verdict, j_model = _judge_rag_answer(
            svc.router, cfg, q["question"], res.answer, res.hits
        )
        judge_model_used = j_model
        if verdict is not None:
            faith_scores.append(verdict.faithfulness_score)
            rel_scores.append(verdict.relevance_score)

        per_question_rows.append(
            {
                "id": q["id"],
                "unanswerable": False,
                "filters": q_filters,
                "ir_metrics": ir,
                "em": em,
                "f1": f1,
                "faithfulness_score": verdict.faithfulness_score if verdict else None,
                "relevance_score": verdict.relevance_score if verdict else None,
                "valid_citations": res.valid_citations,
                "retrieval_latency_ms": res.retrieval_latency_ms,
                "total_latency_ms": res.total_latency_ms,
                "provider_used": res.provider_used,
                "model_used": res.model_used,
            }
        )

    # 3. Chunking ablation & synthetic scale latency benchmark + cost model
    ablation = run_chunking_ablation(corpus_dir, questions, svc.embedder, k=k)
    scale_bench = run_scale_latency_benchmark(svc.embedder) if run_scale_bench else {}
    write_cost_artifacts(out_dir)

    summary: dict[str, Any] = {
        "k": k,
        "total_questions": len(questions),
        "answerable_questions": len(ir_list),
        "unanswerable_questions": unanswerable_total,
        "idempotency_check": {
            "first_ingest": asdict(first_ingest),
            "second_ingest": asdict(second_ingest),
            "zero_duplicate_vectors": (
                first_ingest.lancedb_vector_count == second_ingest.lancedb_vector_count
            ),
        },
        "retrieval_metrics": {
            "recall_at_k": round(statistics.mean(m["recall_at_k"] for m in ir_list), 4),
            "hit_rate_at_k": round(statistics.mean(m["hit_rate"] for m in ir_list), 4),
            "mrr": round(statistics.mean(m["mrr"] for m in ir_list), 4),
            "ndcg_at_k": round(statistics.mean(m["ndcg_at_k"] for m in ir_list), 4),
            "context_precision": round(
                statistics.mean(m["context_precision"] for m in ir_list), 4
            ),
        },
        "answer_quality": {
            "judge_model_used": judge_model_used,
            "mean_faithfulness_1_to_5": (
                round(statistics.mean(faith_scores), 3) if faith_scores else None
            ),
            "mean_relevance_1_to_5": (
                round(statistics.mean(rel_scores), 3) if rel_scores else None
            ),
            "mean_em": round(statistics.mean(em_scores), 4) if em_scores else 0.0,
            "mean_f1": round(statistics.mean(f1_scores), 4) if f1_scores else 0.0,
            "no_context_refusal_accuracy": (
                round(unanswerable_refused / unanswerable_total, 4)
                if unanswerable_total
                else 1.0
            ),
            "unanswerable_refused": unanswerable_refused,
        },
        "latency_ms": {
            "corpus_retrieval_p50_ms": _percentile(retrieval_latencies, 50),
            "corpus_retrieval_p95_ms": _percentile(retrieval_latencies, 95),
            "corpus_e2e_p50_ms": _percentile(e2e_latencies, 50),
            "corpus_e2e_p95_ms": _percentile(e2e_latencies, 95),
        },
        "qdrant_bonus_comparison": {
            "enabled": sync_qdrant,
            "mode": svc._get_qdrant().mode_label if sync_qdrant else None,
            "recall_at_k": (
                round(statistics.mean(m["recall_at_k"] for m in qdrant_ir_list), 4)
                if qdrant_ir_list
                else None
            ),
            "mrr": (
                round(statistics.mean(m["mrr"] for m in qdrant_ir_list), 4)
                if qdrant_ir_list
                else None
            ),
            "retrieval_p50_ms": _percentile(qdrant_latencies, 50) if qdrant_latencies else None,
            "retrieval_p95_ms": _percentile(qdrant_latencies, 95) if qdrant_latencies else None,
        },
        "chunking_ablation": ablation,
        "scale_latency_benchmark": scale_bench,
        "per_question": per_question_rows,
    }

    (out_dir / "p1_eval_results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
