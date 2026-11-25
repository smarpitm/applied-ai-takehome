"""Generate a deterministic multi-format technical corpus (PDF, HTML, MD) for Problem 1.

Each document contains multiple rich technical sections (~700-1,100 tokens per doc) so a
512-token / 64-token-overlap recursive splitter creates realistic multi-chunk splits and
measurable differences across chunking ablation settings.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

CORPUS_DIR = Path(__file__).resolve().parent.parent / "problem1_rag" / "data" / "corpus"

DOCUMENTS: list[dict[str, object]] = [
    # --- Markdown documents (4) ---
    {
        "rel_path": "architecture/vector_storage_overview.md",
        "format": "md",
        "title": "Vector Storage Architecture Overview",
        "topic": "architecture",
        "sections": [
            (
                "Columnar Storage Format",
                (
                    "NimbusStack stores embedding vectors in the disk-native Lance columnar format "
                    "backed by AWS S3 in us-east-1. Each vector uses 384 float32 dimensions "
                    "(1,536 bytes uncompressed) generated locally by BAAI/bge-small-en-v1.5. "
                    "Using 384 dimensions instead of 1,536 dimensions reduces raw vector storage "
                    "footprint and S3 GET bandwidth by 4x at 10 million vectors. Columnar data "
                    "files separate vector columns from payload text so metadata-only scans avoid "
                    "reading vector blobs into page cache. Furthermore, zero-copy Arrow memory "
                    "mapping allows embedded workers to stream slices straight from disk blocks "
                    "without serializing intermediate protobuf frames across network sockets."
                ),
            ),
            (
                "IVF-PQ Partitioning and Memory Budget",
                (
                    "For lightly-queried indexes above 100,000 vectors, NimbusStack enables "
                    "IVF-PQ indexing with 256 partitions and 48 subvectors (8 bits per code), "
                    "compressing each 384-dimensional vector from 1,536 bytes down to 48 bytes. "
                    "Only the matching IVF centroids and requested partitions are fetched on "
                    "demand, avoiding always-on memory-resident HNSW pods. At 10M vectors, raw "
                    "PQ codes occupy only 480 MB total, allowing coarse routing tables under "
                    "4 MB to remain resident in worker RAM while cold partitions rest on S3. "
                    "Operators accept two primary trade-offs: cold partition fetches incur higher "
                    "p95 latency than memory-pinned HNSW graphs, and lossy Product Quantization "
                    "typically trades 1.5% to 3% Recall@5 for roughly 32x vector compression."
                ),
            ),
            (
                "Deterministic Upsert and Fragment Versioning",
                (
                    "Every chunk row receives a deterministic chunk_id computed as the SHA-256 "
                    "digest of normalized source_id concatenated with chunk_index. Before writing "
                    "updated chunks, the ingestor compares the stored doc_hash (SHA-256 of raw "
                    "document text) against the incoming document hash. Unchanged files skip "
                    "re-embedding entirely. Modified files delete prior rows matching source_id "
                    "and then execute LanceDB merge_insert keyed on chunk_id, guaranteeing zero "
                    "duplicate vectors even if a document shrinks in chunk count."
                ),
            ),
            (
                "When to Migrate Back to Managed Vector Databases",
                (
                    "While embedded LanceDB dominates lightly-queried workloads below 5 queries "
                    "per second (QPS), teams should migrate back to a dedicated managed service "
                    "such as Pinecone or Qdrant Cloud when sustained traffic exceeds 50 QPS with "
                    "strict sub-20 ms p99 SLAs, when multi-region active-active write replication "
                    "under 100 ms becomes mandatory, or when streaming updates exceed 1,000 "
                    "vector mutations per second requiring continuous HNSW background indexing."
                ),
            ),
        ],
    },
    {
        "rel_path": "security/iam_zero_trust.md",
        "format": "md",
        "title": "Zero-Trust IAM and mTLS Policy",
        "topic": "security",
        "sections": [
            (
                "Workload Identity and Certificate TTL",
                (
                    "All internal RPC services authenticate using SPIFFE workload identities "
                    "and mutual TLS (mTLS). Short-lived X.509 certificates are issued with a "
                    "strict 24-hour time-to-live (TTL) and rotated automatically at 12 hours "
                    "by the node agent. Static long-lived API tokens are prohibited in "
                    "production clusters. Every workload certificate encodes namespace, cluster "
                    "region, and service role inside Subject Alternative Name (SAN) URIs so "
                    "downstream proxies enforce identity-aware RBAC without shared secrets."
                ),
            ),
            (
                "Audit Trail Retention",
                (
                    "Every authorization decision is emitted as a structured JSON event to an "
                    "immutable append-only sink. Hot audit logs remain searchable for 90 days "
                    "and are archived to S3 Glacier Deep Archive for 7 years to meet SOC 2 "
                    "Type II compliance controls. Log payloads record request_id, caller SPIFFE "
                    "ID, policy version, decision latency, and allow/deny reason while redacting "
                    "bearer credentials and tenant document bodies."
                ),
            ),
            (
                "Break-Glass Emergency Access",
                (
                    "During severity-1 control-plane outages where the primary identity issuer "
                    "is unreachable, two senior SREs must jointly approve a hardware-backed "
                    "YubiKey quorum to mint a 60-minute break-glass role. Break-glass sessions "
                    "trigger real-time PagerDuty notifications and mandatory post-incident "
                    "session replay review within 24 hours."
                ),
            ),
            (
                "Cross-Account Access Boundaries",
                (
                    "Production vector buckets deny cross-account assume-role chains unless "
                    "explicitly whitelisted by AWS Organizations SCPs with external ID and "
                    "source VPC endpoint conditions attached."
                ),
            ),
        ],
    },
    {
        "rel_path": "operations/slo_error_budgets.md",
        "format": "md",
        "title": "SLO Targets and Error Budgets",
        "topic": "operations",
        "sections": [
            (
                "Latency and Availability Objectives",
                (
                    "The NimbusStack Query API enforces a monthly availability SLO of 99.95% "
                    "(21.9 minutes maximum downtime per 30-day window). Warm cache-hit "
                    "retrieval p50 latency target is 18 ms and p95 latency target is 45 ms; "
                    "cold object-store partition reads allow a p95 ceiling of 220 ms. "
                    "End-to-end generation latency budgets depend on the selected LLM router "
                    "tier, targeting under 900 ms p95 for fast hosted models."
                ),
            ),
            (
                "Error Budget Burn Policy",
                (
                    "When error budget burn exceeds 5% over a rolling 6-hour window, feature "
                    "deployments freeze automatically until the on-call reliability engineer "
                    "completes root-cause remediation. Canary rollbacks trigger if 5xx error "
                    "rate exceeds 0.5% for 3 consecutive minutes or if p95 retrieval latency "
                    "breaches 350 ms across 500 sampled queries."
                ),
            ),
            (
                "Health Probes and Graceful Degradation",
                (
                    "Readiness checks verify LanceDB manifest readability and local embedding "
                    "model warm state before accepting traffic on port 8000. If the primary "
                    "hosted LLM API hits HTTP 429 rate limits or times out after 10 seconds, "
                    "the router fails over automatically to the configured local fallback model "
                    "and emits fallback_triggered=true inside structured JSON telemetry."
                ),
            ),
        ],
    },
    {
        "rel_path": "api_reference/idempotency_and_retries.md",
        "format": "md",
        "title": "API Idempotency Keys and Retry Policy",
        "topic": "api_reference",
        "sections": [
            (
                "Idempotency Key Semantics",
                (
                    "Mutating POST endpoints accept an Idempotency-Key HTTP header up to "
                    "128 ASCII characters. Keys remain valid for 24 hours; repeating the "
                    "same key with an identical payload replays the cached 200 or 201 JSON "
                    "response, while reusing a key with a mismatched body hash returns HTTP 409 "
                    "Conflict. Clients should generate UUIDv4 keys per logical action and avoid "
                    "sharing keys across distinct tenant workspaces."
                ),
            ),
            (
                "Exponential Backoff with Full Jitter",
                (
                    "Clients receiving HTTP 429 or 503 must honor the Retry-After response "
                    "header when present, otherwise apply exponential backoff starting at "
                    "250 ms base delay capped at 16 seconds across a maximum of 5 attempts "
                    "with full random jitter. Unbounded tight retry loops are flagged by edge "
                    "WAF rules and temporarily throttled for 60 seconds."
                ),
            ),
            (
                "Request Timeouts and Streaming",
                (
                    "Synchronous /query calls enforce a 30-second server deadline. Requests "
                    "supplying invalid k values outside 1 to 50 or unsupported metadata keys "
                    "fail fast with HTTP 422 validation errors detailing exact field constraints."
                ),
            ),
        ],
    },
    # --- HTML documents (4) ---
    {
        "rel_path": "architecture/hybrid_query_routing.html",
        "format": "html",
        "title": "Hybrid Query Routing and Metadata Filters",
        "topic": "architecture",
        "sections": [
            (
                "Pre-Filtering vs Post-Filtering",
                (
                    "NimbusStack pushes metadata predicates such as file_type and source_id "
                    "directly into the LanceDB columnar scan before scoring vectors. Pushing "
                    "selective filters down avoids wasted distance computations and prevents "
                    "top-k starvation when a filter matches fewer than 5% of rows. Post-filtering "
                    "after global top-k ANN search frequently returns fewer than k chunks or "
                    "zero matching documents when the requested topic represents a small fraction "
                    "of the overall index."
                ),
            ),
            (
                "No-Context Similarity Gate",
                (
                    "Before invoking the downstream generation model, the router verifies "
                    "that at least one retrieved chunk meets the calibrated cosine similarity "
                    "threshold of 0.665 for BAAI/bge-small-en-v1.5. Queries scoring below "
                    "threshold terminate early with a structured no-context refusal and incur "
                    "zero generation tokens. This gate eliminates unsupported hallucinations on "
                    "out-of-domain questions while cutting wasted LLM spend on irrelevant queries."
                ),
            ),
            (
                "Inline Chunk Citation Enforcement",
                (
                    "Every chunk inserted into the prompt context is labeled with its exact "
                    "chunk_id token inside square brackets. The system instruction directs "
                    "the generator to cite only retrieved IDs inline and to reject unsupported "
                    "claims. Post-generation verification flags any cited identifier absent "
                    "from the retrieved candidate list."
                ),
            ),
        ],
    },
    {
        "rel_path": "security/secret_management_guide.html",
        "format": "html",
        "title": "Secret Management and Envelope Encryption",
        "topic": "security",
        "sections": [
            (
                "KMS Envelope Encryption",
                (
                    "Tenant document payloads stored in object storage are encrypted using "
                    "AES-256-GCM data encryption keys (DEKs). Each DEK is wrapped by an AWS "
                    "KMS customer-managed key rotated every 365 days, and plaintext DEKs never "
                    "touch persistent disk. Decryption happens solely in volatile worker memory "
                    "during chunk extraction and query synthesis."
                ),
            ),
            (
                "Secret Scanning and Pre-Commit Gates",
                (
                    "CI pipelines reject pull requests containing high-entropy API tokens or "
                    ".env files. Local developers load credentials exclusively via .env "
                    "files excluded from git and read through pydantic-settings SecretStr fields, "
                    "ensuring API keys never serialize into structured JSON telemetry logs or "
                    "exception stack traces."
                ),
            ),
            (
                "Key Rotation Runbook",
                (
                    "Compromised API credentials must be revoked at the provider console within "
                    "15 minutes and rotated across staging and production secret vaults using "
                    "zero-downtime dual-secret overlap windows."
                ),
            ),
        ],
    },
    {
        "rel_path": "operations/cold_start_mitigation.html",
        "format": "html",
        "title": "Cold-Start Tail Latency Mitigation",
        "topic": "operations",
        "sections": [
            (
                "NVMe Page Cache Warming",
                (
                    "To bound cold p95 retrieval latency on serverless workers, NimbusStack "
                    "pins IVF coarse centroid tables (under 4 MB for 10M vectors) inside an "
                    "ephemeral NVMe read-through cache of 8 GB per worker node. Subsequent "
                    "queries hitting warmed PQ codebooks drop from ~190 ms cold S3 fetch down "
                    "to ~18 ms warm local read."
                ),
            ),
            (
                "Batch Compaction Schedule",
                (
                    "Incremental ingestion writes small fragment files that degrade scan "
                    "throughput if left uncompacted. A nightly compaction job merges fragments "
                    "exceeding 50 deletes or more than 16 delta files into contiguous 256 MB "
                    "Lance segments, keeping read amplification bounded across months of "
                    "incremental document updates."
                ),
            ),
            (
                "Connection Pool Sizing",
                (
                    "Object-store HTTP clients maintain up to 64 keep-alive connections per "
                    "worker so parallel IVF partition range-reads avoid repeated TLS handshakes."
                ),
            ),
        ],
    },
    {
        "rel_path": "api_reference/pagination_and_quotas.html",
        "format": "html",
        "title": "Cursor Pagination and Tenant Rate Limits",
        "topic": "api_reference",
        "sections": [
            (
                "Opaque Cursor Pagination",
                (
                    "List endpoints return opaque base64url next_cursor tokens with a default "
                    "page size of 50 items and a hard upper limit of 200 items per request. "
                    "Offset-based pagination is disabled above 10,000 rows to avoid linear "
                    "table scan degradation. Cursors expire after 60 minutes and bind to the "
                    "caller's original filter predicate hash."
                ),
            ),
            (
                "Token Bucket Quotas",
                (
                    "Free-tier tenants receive 30 queries per minute (burst 10) and 50 MB of "
                    "vector storage. Standard-tier tenants scale to 600 queries per minute "
                    "with bursting up to 100 concurrent requests via token-bucket limiters. "
                    "Response headers X-RateLimit-Remaining and X-RateLimit-Reset expose live "
                    "bucket headroom on every API call."
                ),
            ),
            (
                "Payload Compression",
                (
                    "API responses above 1 KB automatically apply zstd or gzip HTTP content "
                    "encoding when advertised by Accept-Encoding headers."
                ),
            ),
        ],
    },
    # --- PDF documents (4) ---
    {
        "rel_path": "architecture/cost_benchmark_whitepaper.pdf",
        "format": "pdf",
        "title": "Cost Analysis: Disk-Native vs Always-On Pods",
        "topic": "architecture",
        "sections": [
            (
                "Memory Cost Bottleneck in Managed Pods",
                (
                    "Always-on managed vector pods allocate RAM proportional to stored "
                    "vector count regardless of query volume. A 10-million-vector index at "
                    "1,536 float32 dimensions requires roughly 61.4 GB of raw vector memory "
                    "plus HNSW graph overhead (~1.6x), costing upwards of $1,400 to $2,100 "
                    "per month even at 0.2 queries per second. For lightly-queried enterprise "
                    "knowledge bases, paying 24/7 for idle RAM dominates infrastructure bills."
                ),
            ),
            (
                "Disk-Native S3 Economics",
                (
                    "Compressing 384-dim vectors with IVF-PQ (48 bytes/vector) brings 10M "
                    "vectors down to 480 MB of quantized vector data plus ~3.2 GB of text and "
                    "metadata chunks, costing under $0.10/month in S3 storage and roughly "
                    "$18/month on a single t4g.small compute node for low-QPS workloads. "
                    "Even including S3 GET request fees at 0.2 QPS (~520K queries/month), "
                    "object-store egress and request charges remain under $2/month."
                ),
            ),
            (
                "Qdrant On-Disk Memmap Comparison",
                (
                    "Self-hosted Qdrant with on_disk memmap payloads and int8 scalar "
                    "quantization bridges the gap between RAM-heavy HNSW and disk-native "
                    "LanceDB. Scalar quantization reduces vector memory 4x while keeping HNSW "
                    "navigation fast, though running a dedicated Qdrant server still adds "
                    "always-on container orchestration overhead compared to embedded LanceDB."
                ),
            ),
        ],
    },
    {
        "rel_path": "security/pii_redaction_pipeline.pdf",
        "format": "pdf",
        "title": "PII Redaction Before Vector Indexing",
        "topic": "security",
        "sections": [
            (
                "Pre-Embedding Redaction Rules",
                (
                    "Before chunking and embedding, the ingestion pipeline masks Social "
                    "Security Numbers, payment card PANs, and raw bearer tokens with typed "
                    "placeholders such as [REDACTED_SSN] and [REDACTED_CARD]. Redaction runs "
                    "deterministically before computing doc_hash so repeated runs never leak "
                    "sensitive tokens into vector indexes or logs."
                ),
            ),
            (
                "Tenant Deletion SLA",
                (
                    "Right-to-erasure requests under GDPR Article 17 execute immediate "
                    "predicate deletes by tenant_id across active Lance tables and purge "
                    "historical time-travel snapshots within 72 hours. Deletion jobs emit "
                    "signed compliance receipts linking tenant_id to compacted manifest IDs."
                ),
            ),
            (
                "Zero-Retention Inference Providers",
                (
                    "Enterprise configurations disable provider prompt logging via zero-data-"
                    "retention API headers or route sensitive workloads strictly to local "
                    "Ollama models running inside the tenant VPC."
                ),
            ),
        ],
    },
    {
        "rel_path": "operations/backup_and_disaster_recovery.pdf",
        "format": "pdf",
        "title": "Backup, Snapshots, and Disaster Recovery",
        "topic": "operations",
        "sections": [
            (
                "Cross-Region S3 Replication",
                (
                    "Lance commits are atomic manifest versions stored in S3. Cross-region "
                    "replication copies committed manifests and data fragments from us-east-1 "
                    "to us-west-2 with a Recovery Point Objective (RPO) of 15 minutes and "
                    "a Recovery Time Objective (RTO) under 30 minutes. Because manifests are "
                    "immutable, readers can pin historical snapshot versions during rollbacks."
                ),
            ),
            (
                "Quarterly Restore Drills",
                (
                    "Platform SRE runs automated restore verification every 90 days by "
                    "mounting the latest replicated manifest in an isolated staging account "
                    "and validating row counts and top-k recall parity against production."
                ),
            ),
            (
                "Corrupt Fragment Isolation",
                (
                    "Checksum mismatches on object-store reads immediately trigger retries "
                    "against the replica bucket and raise a high-priority integrity alert."
                ),
            ),
        ],
    },
    {
        "rel_path": "api_reference/ingestion_webhook_spec.pdf",
        "format": "pdf",
        "title": "Document Ingestion Webhook Specification",
        "topic": "api_reference",
        "sections": [
            (
                "HMAC-SHA256 Signature Verification",
                (
                    "Outbound ingestion completion webhooks include an X-Nimbus-Signature "
                    "header containing a hex-encoded HMAC-SHA256 digest computed over "
                    "timestamp + '.' + raw_request_body. Receivers must reject signatures "
                    "older than 300 seconds to prevent replay attacks and verify digests using "
                    "constant-time comparison."
                ),
            ),
            (
                "Batch Size and Payload Limits",
                (
                    "Each synchronous POST /ingest batch accepts up to 50 files or 25 MB "
                    "total uncompressed payload. Larger archives must register an S3 prefix "
                    "so asynchronous workers stream chunks without blocking API threads."
                ),
            ),
            (
                "Dead-Letter Queue Semantics",
                (
                    "Webhook endpoints failing 5 consecutive delivery attempts route event "
                    "metadata into a 7-day dead-letter queue inspectable via CLI."
                ),
            ),
        ],
    },
]


def _write_markdown(path: Path, doc: dict[str, object]) -> None:
    lines = [
        "---",
        f"title: {doc['title']}",
        f"topic: {doc['topic']}",
        "---",
        "",
        f"# {doc['title']}",
        "",
    ]
    for heading, body in doc["sections"]:  # type: ignore[attr-defined]
        lines.extend([f"## {heading}", "", body, ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_html(path: Path, doc: dict[str, object]) -> None:
    section_blocks = []
    for heading, body in doc["sections"]:  # type: ignore[attr-defined]
        section_blocks.append(f"<section><h2>{heading}</h2><p>{body}</p></section>")
    html = (
        "<!DOCTYPE html>\n"
        "<html lang='en'>\n"
        f"<head><meta charset='utf-8'><meta name='topic' content='{doc['topic']}'>"
        f"<title>{doc['title']}</title></head>\n"
        "<body>\n"
        "<nav><a href='/'>NimbusStack Docs</a> | <a href='/status'>Status</a></nav>\n"
        f"<main><h1>{doc['title']}</h1>\n"
        + "\n".join(section_blocks)
        + "\n</main>\n<footer>Copyright NimbusStack Engineering</footer>\n</body>\n</html>\n"
    )
    path.write_text(html, encoding="utf-8")


def _write_pdf(path: Path, doc: dict[str, object]) -> None:
    pdf = pymupdf.open()
    for idx, (heading, body) in enumerate(doc["sections"], start=1):  # type: ignore[arg-type]
        page = pdf.new_page()
        header = f"{doc['title']} (Topic: {doc['topic']})" if idx == 1 else str(doc["title"])
        text = f"{header}\n\n{heading}\n\n{body}\n"
        page.insert_textbox(pymupdf.Rect(54, 54, 540, 740), text, fontsize=11, fontname="helv")
    pdf.save(path)
    pdf.close()


def generate_corpus(output_dir: Path = CORPUS_DIR) -> list[Path]:
    """Create all deterministic sample PDF, HTML, and Markdown corpus documents."""
    created: list[Path] = []
    for doc in DOCUMENTS:
        out_path = output_dir / str(doc["rel_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fmt = doc["format"]
        if fmt == "md":
            _write_markdown(out_path, doc)
        elif fmt == "html":
            _write_html(out_path, doc)
        elif fmt == "pdf":
            _write_pdf(out_path, doc)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        created.append(out_path)
    return created


if __name__ == "__main__":
    files = generate_corpus()
    print(f"Generated {len(files)} corpus files under {CORPUS_DIR}")
