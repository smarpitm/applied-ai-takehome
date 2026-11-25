"""Document parsers for PDF (PyMuPDF), HTML (BeautifulSoup4), and Markdown."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pymupdf
from bs4 import BeautifulSoup


@dataclass
class ParsedSection:
    """Logical section extracted from a document."""

    section_header: str
    text: str
    page_number: int | None = None


@dataclass
class ParsedDocument:
    """Normalized representation of a source document ready for chunking."""

    source_id: str
    file_type: Literal["pdf", "html", "md"]
    title: str
    topic: str
    doc_hash: str
    sections: list[ParsedSection]


SUPPORTED_EXTENSIONS: dict[str, Literal["pdf", "html", "md"]] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "md",
    ".markdown": "md",
}


def compute_text_hash(text: str) -> str:
    """Return SHA-256 hex digest of normalized document text for version tracking."""
    normalized = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _infer_topic_from_path(source_id: str) -> str:
    parts = Path(source_id).parts
    return parts[0] if len(parts) > 1 else "general"


def parse_markdown(path: Path, root_dir: Path) -> ParsedDocument:
    """Parse Markdown file with optional YAML frontmatter and heading sections."""
    source_id = path.relative_to(root_dir).as_posix()
    raw = path.read_text(encoding="utf-8")
    topic = _infer_topic_from_path(source_id)
    title = path.stem.replace("_", " ").title()

    body = raw
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", raw, flags=re.DOTALL)
    if fm_match:
        fm_block = fm_match.group(1)
        body = raw[fm_match.end() :]
        for line in fm_block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                if key.strip().lower() == "topic" and val.strip():
                    topic = val.strip()
                elif key.strip().lower() == "title" and val.strip():
                    title = val.strip()

    sections: list[ParsedSection] = []
    current_header = title
    current_lines: list[str] = []

    for line in body.splitlines():
        heading_match = re.match(r"^#{1,3}\s+(.+)$", line.strip())
        if heading_match:
            content = "\n".join(current_lines).strip()
            if content:
                sections.append(ParsedSection(section_header=current_header, text=content))
            current_header = heading_match.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)

    tail = "\n".join(current_lines).strip()
    if tail:
        sections.append(ParsedSection(section_header=current_header, text=tail))

    if not sections and body.strip():
        sections.append(ParsedSection(section_header=title, text=body.strip()))

    full_text = "\n\n".join(f"{s.section_header}\n{s.text}" for s in sections)
    return ParsedDocument(
        source_id=source_id,
        file_type="md",
        title=title,
        topic=topic,
        doc_hash=compute_text_hash(full_text),
        sections=sections,
    )


def parse_html(path: Path, root_dir: Path) -> ParsedDocument:
    """Parse HTML file using BeautifulSoup4, stripping navigation/footer boilerplate."""
    source_id = path.relative_to(root_dir).as_posix()
    raw = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "html.parser")

    topic = _infer_topic_from_path(source_id)
    meta_topic = soup.find("meta", attrs={"name": "topic"})
    if meta_topic and meta_topic.get("content"):
        topic = str(meta_topic["content"]).strip()

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else path.stem.replace("_", " ").title()

    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()

    main_node = soup.find("main") or soup.body or soup
    sections: list[ParsedSection] = []
    current_header = title
    current_chunks: list[str] = []

    for elem in main_node.find_all(["h1", "h2", "h3", "p", "li", "pre"]):
        text = re.sub(r"\s+", " ", elem.get_text(" ", strip=True)).strip()
        if not text:
            continue
        if elem.name in {"h1", "h2", "h3"}:
            if current_chunks:
                sections.append(
                    ParsedSection(section_header=current_header, text="\n".join(current_chunks))
                )
                current_chunks = []
            current_header = text
        else:
            current_chunks.append(text)

    if current_chunks:
        sections.append(
            ParsedSection(section_header=current_header, text="\n".join(current_chunks))
        )

    if not sections:
        fallback = re.sub(r"\s+", " ", main_node.get_text(" ", strip=True)).strip()
        if fallback:
            sections.append(ParsedSection(section_header=title, text=fallback))

    full_text = "\n\n".join(f"{s.section_header}\n{s.text}" for s in sections)
    return ParsedDocument(
        source_id=source_id,
        file_type="html",
        title=title,
        topic=topic,
        doc_hash=compute_text_hash(full_text),
        sections=sections,
    )


def parse_pdf(path: Path, root_dir: Path) -> ParsedDocument:
    """Parse PDF file page-by-page using PyMuPDF (fitz)."""
    source_id = path.relative_to(root_dir).as_posix()
    topic = _infer_topic_from_path(source_id)
    title = path.stem.replace("_", " ").title()

    sections: list[ParsedSection] = []
    with pymupdf.open(path) as pdf:
        for page_idx, page in enumerate(pdf, start=1):
            raw_text = page.get_text("text").strip()
            if not raw_text:
                continue
            lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
            section_header = f"page_{page_idx}"
            if len(lines) >= 2:
                # Use the first short heading-like line on the page if available
                candidate = lines[1] if page_idx == 1 and len(lines) > 2 else lines[0]
                if len(candidate) <= 90:
                    section_header = candidate
            cleaned = re.sub(r"\s+", " ", raw_text).strip()
            sections.append(
                ParsedSection(section_header=section_header, text=cleaned, page_number=page_idx)
            )

    full_text = "\n\n".join(f"{s.section_header}\n{s.text}" for s in sections)
    return ParsedDocument(
        source_id=source_id,
        file_type="pdf",
        title=title,
        topic=topic,
        doc_hash=compute_text_hash(full_text),
        sections=sections,
    )


def parse_document(path: Path, root_dir: Path) -> ParsedDocument:
    """Dispatch document parser based on file extension."""
    ext = path.suffix.lower()
    fmt = SUPPORTED_EXTENSIONS.get(ext)
    if fmt == "md":
        return parse_markdown(path, root_dir)
    if fmt == "html":
        return parse_html(path, root_dir)
    if fmt == "pdf":
        return parse_pdf(path, root_dir)
    raise ValueError(f"Unsupported file extension '{ext}' for {path}")


def discover_corpus_files(corpus_dir: Path) -> list[Path]:
    """Return sorted list of supported PDF/HTML/MD files under corpus_dir."""
    if not corpus_dir.exists():
        return []
    files = [
        p
        for p in corpus_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.relative_to(corpus_dir).as_posix())
