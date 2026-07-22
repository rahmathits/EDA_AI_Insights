"""
Very small in-memory session store.

Each uploaded dataset gets a session_id (uuid4). We keep the dataframe and
per-agent results in memory, keyed by session_id. This is intentionally
simple (no external DB) -- swap for Redis/Postgres if you need persistence
across restarts or multi-worker deployments.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


@dataclass
class Session:
    session_id: str
    filename: str
    df: pd.DataFrame
    results: Dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False
    status: str = "uploaded"  # uploaded -> running -> complete/cancelled/error
    llm_api_key: Optional[str] = None      # runtime, session-scoped -- never persisted to disk/.env
    business_context: Optional[str] = None  # free text describing the business case, used by the summary prompt


class SessionStore:
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def create(self, filename: str, df: pd.DataFrame) -> Session:
        session_id = str(uuid.uuid4())
        session = Session(session_id=session_id, filename=filename, df=df)
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def require(self, session_id: str) -> Session:
        session = self.get(session_id)
        if session is None:
            raise KeyError(f"Unknown session_id: {session_id}")
        return session

    def delete(self, session_id: str):
        with self._lock:
            self._sessions.pop(session_id, None)


store = SessionStore()