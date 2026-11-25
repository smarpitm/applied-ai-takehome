"""CLI entrypoint for Problem 1 Cost-Efficient RAG Service."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from problem1_rag.cost_model import generate_cost_comparison, write_cost_artifacts
from problem1_rag.eval_harness import run_full_evaluation
from problem1_rag.rag_pipeline import RAGService

app = typer.Typer(help="Problem 1: Cost-Efficient RAG CLI (LanceDB + Qdrant)")
console = Console()


@app.command()
def ingest(
    corpus_dir: Annotated[
        Path, typer.Option(help="Directory of PDF/HTML/MD files")
    ] = Path("problem1_rag/data/corpus"),
    chunk_size: Annotated[int, typer.Option(help="Chunk size in tokens")] = 512,
    chunk_overlap: Annotated[int, typer.Option(help="Chunk overlap in tokens")] = 64,
    sync_qdrant: Annotated[bool, typer.Option(help="Also ingest into Qdrant bonus store")] = False,
) -> None:
    """Idempotently ingest a document folder into LanceDB (and optionally Qdrant)."""
    svc = RAGService()
    summary = svc.ingest_directory(
        corpus_dir=corpus_dir,
        chunk_size_tokens=chunk_size,
        chunk_overlap_tokens=chunk_overlap,
        sync_qdrant=sync_qdrant,
    )
    console.print_json(json.dumps(asdict(summary)))


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer over indexed docs")],
    k: Annotated[int, typer.Option(help="Top-k chunks to retrieve")] = 5,
    file_type: Annotated[str | None, typer.Option(help="Filter by file_type (pdf|html|md)")] = None,
    topic: Annotated[str | None, typer.Option(help="Filter by topic metadata")] = None,
    store: Annotated[str, typer.Option(help="Vector store: lancedb or qdrant")] = "lancedb",
) -> None:
    """Query the RAG service with optional metadata filters and print cited answer."""
    filters: dict[str, str] = {}
    if file_type:
        filters["file_type"] = file_type
    if topic:
        filters["topic"] = topic

    svc = RAGService()
    res = svc.query(question, k=k, filters=filters or None, store=store)
    console.print(f"[bold]Answer ({res.provider_used}/{res.model_used}):[/bold] {res.answer}")
    console.print(
        f"[dim]no_context={res.no_context} | chunks={res.returned_chunk_count} | "
        f"citations={res.valid_citations} | retrieval_ms={res.retrieval_latency_ms} | "
        f"llm_ms={res.llm_latency_ms} | tokens={res.input_tokens}/{res.output_tokens}[/dim]"
    )


@app.command()
def stats() -> None:
    """Display LanceDB store statistics."""
    svc = RAGService()
    console.print_json(json.dumps(svc.lance_store.stats()))


@app.command("cost-table")
def cost_table(
    out_dir: Annotated[
        Path, typer.Option(help="Output directory for cost files")
    ] = Path("results"),
) -> None:
    """Generate and print the 100K / 1M / 10M monthly cost comparison table."""
    rows, _ = generate_cost_comparison()
    write_cost_artifacts(out_dir)
    table = Table(title="Monthly Cost Across Scale (384-dim, 0.2 QPS)")
    table.add_column("Scale")
    table.add_column("LanceDB S3+Compute ($/mo)", justify="right")
    table.add_column("Qdrant Self-Hosted ($/mo)", justify="right")
    table.add_column("Pinecone Serverless ($/mo)", justify="right")
    table.add_column("Pinecone Pod ($/mo)", justify="right")
    table.add_column("Savings vs Pod", justify="right")
    for r in rows:
        table.add_row(
            r.label,
            f"${r.lancedb_total_usd:.2f}",
            f"${r.qdrant_self_hosted_usd:.2f}",
            f"${r.pinecone_serverless_usd:.2f}",
            f"${r.pinecone_pod_usd:.2f}",
            f"{r.savings_vs_pinecone_pod_pct:.1f}%",
        )
    console.print(table)
    console.print(f"Saved cost artifacts to {out_dir}")


@app.command("eval")
def evaluate(
    corpus_dir: Annotated[Path, typer.Option()] = Path("problem1_rag/data/corpus"),
    questions_path: Annotated[
        Path, typer.Option()
    ] = Path("problem1_rag/data/eval_questions.json"),
    out_dir: Annotated[Path, typer.Option()] = Path("results"),
    k: Annotated[int, typer.Option()] = 5,
    sync_qdrant: Annotated[bool, typer.Option()] = True,
    scale_bench: Annotated[bool, typer.Option()] = True,
) -> None:
    """Run the full 3-layer RAG evaluation harness."""
    summary = run_full_evaluation(
        corpus_dir=corpus_dir,
        questions_path=questions_path,
        out_dir=out_dir,
        k=k,
        sync_qdrant=sync_qdrant,
        run_scale_bench=scale_bench,
    )
    console.print_json(json.dumps({k2: v for k2, v in summary.items() if k2 != "per_question"}))


if __name__ == "__main__":
    app()
