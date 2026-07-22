from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    n_rows: int
    n_cols: int
    columns: List[str]


class StepResult(BaseModel):
    agent: str
    status: str
    duration_sec: float
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class PipelineStatus(BaseModel):
    session_id: str
    status: str
    completed_steps: List[str]


class ErrorResponse(BaseModel):
    detail: str


class ApiKeyRequest(BaseModel):
    api_key: str


class ApiKeyResponse(BaseModel):
    valid: bool
    message: str
    llm_available: bool


class LlmStatusResponse(BaseModel):
    llm_available: bool
    source: str  # "session" | "env" | "none"
    llm_model: str


class BusinessContextRequest(BaseModel):
    business_context: str