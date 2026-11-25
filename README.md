# Applied AI / ML Engineering — Take-Home Assignment

> **Status:** Repository scaffold initialized. Core services, evaluation harnesses, and benchmark tables are staged for build approval.

This repository implements both problems from [docs/Gen AI_assignment.pdf](docs/Gen%20AI_assignment.pdf) alongside deliverables aligned with [docs/submission_template (1).docx](docs/submission_template%20(1).docx):

1. **Problem 1 — Cost-Efficient RAG Application (`problem1_rag/`)**
   - **Primary Vector Store:** Embedded **LanceDB** (disk-native Lance columnar format, IVF-PQ partitioning, SQL-style metadata filtering, and idempotent `merge_insert` upserts).
   - **Bonus Benchmark Store:** Self-hosted **Qdrant** (`docker-compose.yml` with on-disk memmap + `int8` scalar quantization, plus local disk client support).
   - **Ingestion & Embeddings:** `PyMuPDF`, `BeautifulSoup4`, and Markdown loaders; recursive token splitter (`512` tokens / `64` overlap); local 384-dim `BAAI/bge-small-en-v1.5` via FastEmbed (4x storage/I/O reduction vs. 1536-dim).
   - **Service & Guardrails:** FastAPI service + CLI (`pydantic-settings`), structured JSON request logging, similarity-threshold refusal prior to LLM invocation, and inline `[chunk_id]` citations via low-cost hosted inference (Groq / Gemini Flash).
   - **Evaluation & Cost Analysis:** 24-question fixed evaluation harness (including unanswerable refusal tests) measuring Retrieval (`Recall@k`, `MRR`, `nDCG@k`, `Context Precision`), Answer Quality (separate strict-JSON LLM judge + `EM`/`F1`), `p50`/`p95` retrieval latency, and `100K / 1M / 10M` vector monthly TCO vs. Pinecone.

2. **Problem 2 — LLM-as-Judge Evaluation Pipeline (`problem2_judge/`)**
   - **Judging Modes:** Pairwise A-vs-B and Pointwise anchored rubric scoring (reference-based & reference-free).
   - **Robustness & Auditability:** Multi-stage JSON verdict recovery, immutable JSONL prompt/response audit logs, token/cost tracking, and independent cross-family generator vs. judge configuration.
   - **Bias Mitigation & Validation:** Code-level mitigations and before/after measurements for Position (A/B order swap), Verbosity/Length, Self-Enhancement, Sycophancy/Style, and Score Clustering, validated via gold agreement (Cohen's $\kappa$), test-retest consistency, adversarial probes, and A/B config winner selection.

---

## Quickstart (Local Setup)

### Prerequisites
- Python **3.11+** (`uv` or `pip`)
- Optional: Docker Desktop for the Qdrant server bonus benchmark (`docker compose up -d`)

### Environment Setup
```powershell
# 1. Clone the repository
git clone https://github.com/smarpitm/applied-ai-takehome.git
cd applied-ai-takehome

# 2. Create and activate the virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Copy environment template and add your API keys
copy .env.example .env

# 4. Install dependencies (once modules are built)
pip install -e ".[dev]"
```

*(On Linux/macOS, activate with `source .venv/bin/activate` and `cp .env.example .env`.)*

---

## Repository Structure

```text
applied-ai-takehome/
├── docs/                   # Assignment prompt PDF & submission template DOCX
├── shared/                 # Shared env config, unified LLM clients, JSON telemetry logger
├── problem1_rag/           # FastAPI RAG service, LanceDB/Qdrant stores, 3-layer eval & cost model
├── problem2_judge/         # Bias-aware LLM-as-Judge pipeline, validation & A/B comparator
├── results/                # Reproducible eval results, cost tables & audit logs
├── scripts/                # Corpus & utility scripts
├── tests/                  # Offline pytest verification suite
├── docker-compose.yml      # Optional self-hosted Qdrant (on-disk memmap)
├── .env.example            # Environment variable reference (no secrets)
└── pyproject.toml          # Dependencies, pytest & ruff configuration
```
