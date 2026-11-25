"""Reproducible monthly cost comparison across 100K, 1M, and 10M vectors."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class CostAssumptions:
    """Explicit infrastructure and workload assumptions for cost modeling."""

    embedding_dim: int = 384
    float32_bytes: int = 4
    pq_bytes_per_vector: int = 48  # 48 subvectors x 8-bit codes
    avg_payload_bytes_compressed: int = 720  # ~1.8 KB text+metadata compressed ~2.5x in Lance
    ivf_centroid_overhead_bytes_per_vec: int = 4  # coarse partitions + Lance manifest overhead
    s3_storage_usd_per_gb_month: float = 0.023  # AWS S3 Standard us-east-1
    s3_get_usd_per_1k: float = 0.0004
    qps: float = 0.2  # lightly-queried index (~518,400 queries/month)
    s3_gets_per_cold_query: float = 1.5  # amortized warm NVMe cache + cold IVF partition read
    lancedb_compute_usd_month_small: float = 15.18  # AWS t4g.small (2 vCPU, 2 GiB RAM) on-demand
    lancedb_compute_usd_month_medium: float = 30.37  # AWS t4g.medium (2 vCPU, 4 GiB RAM) at 10M
    qdrant_ebs_gp3_usd_per_gb_month: float = 0.08
    pinecone_serverless_storage_usd_per_gb_month: float = 0.33
    pinecone_serverless_ru_usd_per_1m: float = 8.25
    pinecone_pod_usd_month: float = 70.0  # s1.x1 / p1.x1 ~ $0.096/hr * 730h = ~$70/pod/mo


@dataclass(frozen=True)
class ScaleCostRow:
    """Monthly cost breakdown for a given vector count."""

    vector_count: int
    label: str
    raw_vectors_gb: float
    raw_1536d_vectors_gb: float
    lancedb_total_stored_gb: float
    lancedb_s3_storage_usd: float
    lancedb_s3_requests_usd: float
    lancedb_compute_usd: float
    lancedb_total_usd: float
    qdrant_self_hosted_usd: float
    pinecone_serverless_usd: float
    pinecone_pod_usd: float
    savings_vs_pinecone_pod_pct: float
    savings_vs_pinecone_serverless_pct: float


def _gb(num_bytes: float) -> float:
    return num_bytes / (1024**3)


def compute_cost_row(vector_count: int, label: str, a: CostAssumptions) -> ScaleCostRow:
    """Compute monthly cost across LanceDB, self-hosted Qdrant, and Pinecone tiers."""
    raw_vec_bytes = vector_count * a.embedding_dim * a.float32_bytes
    raw_1536_bytes = vector_count * 1536 * a.float32_bytes

    # LanceDB: IVF-PQ vectors + compressed payload/metadata + centroid/manifest overhead
    lance_bytes = vector_count * (
        a.pq_bytes_per_vector
        + a.avg_payload_bytes_compressed
        + a.ivf_centroid_overhead_bytes_per_vec
    )
    lance_gb = _gb(lance_bytes)
    s3_storage = lance_gb * a.s3_storage_usd_per_gb_month

    monthly_queries = a.qps * 3600 * 24 * 30
    s3_gets = monthly_queries * a.s3_gets_per_cold_query
    s3_requests = (s3_gets / 1000.0) * a.s3_get_usd_per_1k
    lance_compute = (
        a.lancedb_compute_usd_month_medium
        if vector_count >= 10_000_000
        else a.lancedb_compute_usd_month_small
    )
    lancedb_total = s3_storage + s3_requests + lance_compute

    # Self-hosted Qdrant (Docker on-disk memmap + int8 scalar quantization: 1 byte/dim + payload)
    qdrant_bytes = vector_count * (a.embedding_dim + a.avg_payload_bytes_compressed + 64)
    qdrant_gb = max(10.0, math.ceil(_gb(qdrant_bytes) * 1.25))
    qdrant_compute = 30.37 if vector_count < 10_000_000 else 60.74  # t4g.medium -> t4g.large
    qdrant_total = qdrant_compute + qdrant_gb * a.qdrant_ebs_gp3_usd_per_gb_month

    # Pinecone Serverless (uncompressed float32 + metadata stored, plus Read Units)
    pc_stored_gb = _gb(raw_vec_bytes + vector_count * a.avg_payload_bytes_compressed)
    # Approximate RUs scaling gently with namespace/partition scan size (~10 to ~120 RUs/query)
    rus_per_query = 10 if vector_count <= 100_000 else (30 if vector_count <= 1_000_000 else 120)
    pc_ru_cost = (
        (monthly_queries * rus_per_query) / 1_000_000.0
    ) * a.pinecone_serverless_ru_usd_per_1m
    pc_storage_cost = max(0.01, pc_stored_gb * a.pinecone_serverless_storage_usd_per_gb_month)
    pinecone_serverless = pc_storage_cost + pc_ru_cost

    # Pinecone Pod-based (always-on RAM pods: ~1M 384-dim vectors per s1/p1 pod; 10 pods at 10M)
    pods_needed = max(1, math.ceil(vector_count / 1_000_000))
    pinecone_pod = pods_needed * a.pinecone_pod_usd_month

    sav_pod = ((pinecone_pod - lancedb_total) / pinecone_pod) * 100.0
    sav_sl = ((pinecone_serverless - lancedb_total) / pinecone_serverless) * 100.0

    return ScaleCostRow(
        vector_count=vector_count,
        label=label,
        raw_vectors_gb=round(_gb(raw_vec_bytes), 3),
        raw_1536d_vectors_gb=round(_gb(raw_1536_bytes), 3),
        lancedb_total_stored_gb=round(lance_gb, 3),
        lancedb_s3_storage_usd=round(s3_storage, 3),
        lancedb_s3_requests_usd=round(s3_requests, 3),
        lancedb_compute_usd=round(lance_compute, 2),
        lancedb_total_usd=round(lancedb_total, 2),
        qdrant_self_hosted_usd=round(qdrant_total, 2),
        pinecone_serverless_usd=round(pinecone_serverless, 2),
        pinecone_pod_usd=round(pinecone_pod, 2),
        savings_vs_pinecone_pod_pct=round(sav_pod, 1),
        savings_vs_pinecone_serverless_pct=round(sav_sl, 1),
    )


def generate_cost_comparison(
    assumptions: CostAssumptions | None = None,
) -> tuple[list[ScaleCostRow], str]:
    """Compute 100K / 1M / 10M cost table and render Markdown report."""
    a = assumptions or CostAssumptions()
    rows = [
        compute_cost_row(100_000, "100K", a),
        compute_cost_row(1_000_000, "1M", a),
        compute_cost_row(10_000_000, "10M", a),
    ]

    lines = [
        "# Problem 1 — Monthly Cost Comparison Across Scale (100K / 1M / 10M Vectors)",
        "",
        "| Vectors | Raw 384d (GB) | LanceDB Stored (GB) | LanceDB S3+Compute ($/mo) "
        "| Self-Hosted Qdrant ($/mo) | Pinecone Serverless ($/mo) | Pinecone Pod ($/mo) "
        "| Savings vs Pod |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r.label} | {r.raw_vectors_gb:.3f} | {r.lancedb_total_stored_gb:.3f} | "
            f"${r.lancedb_total_usd:.2f} | ${r.qdrant_self_hosted_usd:.2f} | "
            f"${r.pinecone_serverless_usd:.2f} | ${r.pinecone_pod_usd:.2f} | "
            f"{r.savings_vs_pinecone_pod_pct:.1f}% |"
        )

    lines.extend(
        [
            "",
            "## Explicit Pricing & Workload Assumptions",
            f"- **Embeddings:** `BAAI/bge-small-en-v1.5`, `{a.embedding_dim}` float32 dims "
            f"(`{a.embedding_dim * a.float32_bytes}` bytes/vector raw; 4x smaller than "
            "1536-dim `6,144` bytes).",
            f"- **LanceDB Compression & Payload:** IVF-PQ at `{a.pq_bytes_per_vector}` "
            f"bytes/vector + `{a.avg_payload_bytes_compressed}` bytes compressed text/metadata "
            f"+ `{a.ivf_centroid_overhead_bytes_per_vec}` bytes index overhead.",
            f"- **Workload:** Lightly-queried index at `{a.qps}` QPS (~`518,400` queries/month, "
            f"`{a.s3_gets_per_cold_query}` S3 GETs/query at `${a.s3_get_usd_per_1k}/1K` GETs, "
            f"S3 Standard at `${a.s3_storage_usd_per_gb_month}/GB-mo`).",
            f"- **Compute:** AWS `t4g.small` (`${a.lancedb_compute_usd_month_small}/mo`) up to "
            f"1M vectors, `t4g.medium` (`${a.lancedb_compute_usd_month_medium}/mo`) at 10M vectors "
            "(zero idle RAM scaling per vector; scaling down to serverless Lambda/Fargate cuts "
            "low-QPS idle compute further).",
            f"- **Managed DB Comparison:** Pinecone Pod (`s1.x1` / `p1.x1` at "
            f"`${a.pinecone_pod_usd_month}/pod/mo`, ~1M 384-dim vectors/pod) and Pinecone "
            f"Serverless (`${a.pinecone_serverless_storage_usd_per_gb_month}/GB-mo` + "
            f"`${a.pinecone_serverless_ru_usd_per_1m}/1M RUs`).",
            "- **Self-Hosted Qdrant Bonus:** Dedicated EC2 (`t4g.medium`/`t4g.large`) + EBS "
            "`gp3` (`$0.08/GB-mo`) with `on_disk` memmap + `int8` scalar quantization.",
        ]
    )
    return rows, "\n".join(lines) + "\n"


def write_cost_artifacts(out_dir: Path) -> None:
    """Write reproducible JSON and Markdown cost comparison artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assumptions = CostAssumptions()
    rows, markdown = generate_cost_comparison(assumptions)
    (out_dir / "p1_cost_comparison.md").write_text(markdown, encoding="utf-8")
    payload = {"assumptions": asdict(assumptions), "rows": [asdict(r) for r in rows]}
    (out_dir / "p1_cost_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
