"""
Centralized structured logging for the EDA multi-agent system.

Every event is emitted both to a rotating log file (JSON lines, easy to grep /
ship to a log aggregator) and to an in-memory ring buffer that the API exposes
via GET /logs/{session_id} so the frontend can show a live activity feed.

Captured event types (as required by the spec):
    upload_failure, validation_failure, agent_start, agent_complete,
    memory_usage, dataset_size, warning, retry_attempt,
    statistical_failure, report_generation_failure, user_cancellation
"""
from __future__ import annotations

import json
import logging
import os
import time
import tracemalloc
from collections import deque
from logging.handlers import RotatingFileHandler
from typing import Any, Deque, Dict

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "eda_app.log")

_raw_logger = logging.getLogger("eda_app")
_raw_logger.setLevel(logging.DEBUG)
if not _raw_logger.handlers:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5_000_000, backupCount=3)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _raw_logger.addHandler(handler)
    stream = logging.StreamHandler()
    stream.setFormatter(logging.Formatter("%(message)s"))
    _raw_logger.addHandler(stream)

# session_id -> deque of recent structured events (for the live activity feed)
_EVENT_BUFFERS: Dict[str, Deque[dict]] = {}
_MAX_EVENTS_PER_SESSION = 500


def _buffer_for(session_id: str) -> Deque[dict]:
    if session_id not in _EVENT_BUFFERS:
        _EVENT_BUFFERS[session_id] = deque(maxlen=_MAX_EVENTS_PER_SESSION)
    return _EVENT_BUFFERS[session_id]


def get_events(session_id: str) -> list[dict]:
    return list(_buffer_for(session_id))


def log_event(event_type: str, session_id: str = "-", **fields: Any) -> dict:
    """Emit one structured event. Returns the event dict (useful for tests)."""
    event = {
        "ts": time.time(),
        "event": event_type,
        "session_id": session_id,
        **fields,
    }
    _raw_logger.info(json.dumps(event, default=str))
    _buffer_for(session_id).append(event)
    return event


# ---- convenience wrappers, one per required event category ----------------

def log_upload_failure(session_id: str, filename: str, reason: str):
    return log_event("upload_failure", session_id, filename=filename, reason=reason)


def log_validation_failure(session_id: str, filename: str, reason: str):
    return log_event("validation_failure", session_id, filename=filename, reason=reason)


def log_agent_start(session_id: str, agent: str):
    return log_event("agent_start", session_id, agent=agent)


def log_agent_complete(session_id: str, agent: str, duration_sec: float):
    return log_event("agent_complete", session_id, agent=agent, duration_sec=round(duration_sec, 4))


def log_memory_usage(session_id: str, agent: str, current_mb: float, peak_mb: float):
    return log_event("memory_usage", session_id, agent=agent,
                      current_mb=round(current_mb, 3), peak_mb=round(peak_mb, 3))


def log_dataset_size(session_id: str, rows: int, cols: int, bytes_in_memory: int):
    return log_event("dataset_size", session_id, rows=rows, cols=cols,
                      bytes_in_memory=bytes_in_memory)


def log_warning(session_id: str, agent: str, message: str):
    return log_event("warning", session_id, agent=agent, message=message)


def log_retry_attempt(session_id: str, agent: str, attempt: int, reason: str):
    return log_event("retry_attempt", session_id, agent=agent, attempt=attempt, reason=reason)


def log_statistical_failure(session_id: str, agent: str, test: str, reason: str):
    return log_event("statistical_failure", session_id, agent=agent, test=test, reason=reason)


def log_report_generation_failure(session_id: str, reason: str):
    return log_event("report_generation_failure", session_id, reason=reason)


def log_user_cancellation(session_id: str, stage: str):
    return log_event("user_cancellation", session_id, stage=stage)


class MemoryTracker:
    """Small helper around tracemalloc to sample per-agent memory usage."""

    def __enter__(self):
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self._started_here = not tracemalloc.is_tracing()
        tracemalloc.reset_peak()
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def sample(self):
        current, peak = tracemalloc.get_traced_memory()
        return current / (1024 * 1024), peak / (1024 * 1024)
