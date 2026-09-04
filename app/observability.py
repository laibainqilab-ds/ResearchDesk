"""Local, structured observability for ResearchDesk.

Every RAG request gets a trace_id (a plain UUID4 hex string) generated at
the start of the request and threaded through the pipeline stages that
handle it (query rewriting, multi-query generation, retrieval, reranking,
evidence selection, answer generation, citation validation). Each stage
logs one structured JSON event tagged with that trace_id via `log_event()`,
so every log line belonging to one request can be found with a single
`grep <trace_id> logs/researchdesk.jsonl`.

This is intentionally the simplest thing that works: Python's standard
`logging` module, one local JSON-lines file, no distributed tracing, no
external services, no new database. `configure_logging()` is idempotent
and safe to call from `RAG.__init__()` on every app/process start.
"""

import json
import logging
import time
import uuid
from pathlib import Path

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "researchdesk.jsonl"

LOGGER_NAME = "researchdesk"

_logger = logging.getLogger(LOGGER_NAME)
_configured = False


def configure_logging(log_file: Path = LOG_FILE, level: int = logging.INFO) -> logging.Logger:
    """Attach a local JSON-lines file handler to the researchdesk logger.

    Idempotent -- safe to call on every RAG() construction / Streamlit
    rerun without stacking duplicate handlers or duplicate log lines.
    """
    global _configured

    if _configured:
        return _logger

    log_file.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    _logger.addHandler(handler)
    _logger.setLevel(level)
    # Keep researchdesk's structured JSON lines in their own file only --
    # don't also let them bubble up to the root logger's default handler.
    _logger.propagate = False

    _configured = True
    return _logger


def new_trace_id() -> str:
    """A short, unique identifier for one RAG request or ingestion
    operation. Plain UUID4 -- no distributed-tracing infrastructure."""
    return uuid.uuid4().hex


def log_event(trace_id: str | None, event: str, level: int = logging.INFO, **fields) -> None:
    """Emit one structured JSON log line: {timestamp, trace_id, event,
    level, ...fields}.

    `fields` must never include secrets, API keys, or credentials -- only
    pass request/pipeline metadata (query text, chunk ids, scores, error
    messages, latency, etc.).
    """
    payload = {
        "timestamp": time.time(),
        "trace_id": trace_id,
        "event": event,
        "level": logging.getLevelName(level),
        **fields,
    }

    _logger.log(level, json.dumps(payload, default=str, ensure_ascii=False))
