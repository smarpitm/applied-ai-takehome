"""Append-only JSONL audit logging and structured console telemetry."""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_WRITE_LOCK = threading.Lock()


def utc_now_iso() -> str:
    """Return current UTC ISO-8601 timestamp."""
    return datetime.now(UTC).isoformat()


def append_jsonl(path: Path, record: dict[str, Any] | BaseModel) -> None:
    """Append a single JSON record to an audit JSONL file in a thread-safe way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
    line = json.dumps(payload, ensure_ascii=False)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load all JSON records from an existing JSONL trace file."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def get_json_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Return a logger configured to emit structured single-line JSON messages."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_json_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    """Emit a structured JSON log line."""
    record = {"timestamp": utc_now_iso(), "event": event, **fields}
    logger.info(json.dumps(record, ensure_ascii=False, default=str))
