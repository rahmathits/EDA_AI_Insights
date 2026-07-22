from __future__ import annotations

import io
import os
import time

from dotenv import load_dotenv

load_dotenv()  # must run before llm_helper is imported, since it reads env vars at import time

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import llm_helper
from agents import AGENT_REGISTRY, PIPELINE
from logger_config import (
    get_events,
    log_dataset_size,
    log_upload_failure,
    log_validation_failure,
)
from orchestrator import run_full_pipeline, run_single_step
from schemas import (
    ApiKeyRequest,
    ApiKeyResponse,
    BusinessContextRequest,
    LlmStatusResponse,
    PipelineStatus,
    StepResult,
    UploadResponse,
)
from session_store import store

APP_TITLE = "Advanced EDA Multi-Agent System"
MAX_FILE_MB = 100
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv"}

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware, allow_origins=["https://eda-ai-insights-frontend.vercel.app"], allow_methods=["*"], allow_headers=["*"],
)

STEP_ORDER = [cls().name for cls in PIPELINE]
_last_key_attempt: dict[str, float] = {}  # session_id -> unix time, throttles /llm/connect
KEY_ATTEMPT_COOLDOWN_SEC = 3

# for the root endpoint, we don't want to expose the full API docs, so we mount a static index.html instead

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "EDA AI Insights API is running"
    }


# --------------------------------------------------------------------------
# Upload & validation
# --------------------------------------------------------------------------
@app.post("/api/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        log_validation_failure("-", filename, f"unsupported extension '{ext}'")
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}")

    raw = await file.read()
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        log_upload_failure("-", filename, f"file too large: {size_mb:.1f}MB > {MAX_FILE_MB}MB")
        raise HTTPException(400, f"File exceeds max size of {MAX_FILE_MB}MB")
    if size_mb == 0:
        log_upload_failure("-", filename, "empty file")
        raise HTTPException(400, "Uploaded file is empty")

    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(raw))
        elif ext == ".tsv":
            df = pd.read_csv(io.BytesIO(raw), sep="\t")
        else:
            df = pd.read_excel(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        log_upload_failure("-", filename, f"parse error: {exc}")
        raise HTTPException(400, f"Could not parse file: {exc}") from exc

    if df.empty or df.shape[1] == 0:
        log_validation_failure("-", filename, "no rows/columns after parsing")
        raise HTTPException(400, "Dataset has no usable rows or columns")

    session = store.create(filename, df)
    log_dataset_size(session.session_id, df.shape[0], df.shape[1], int(df.memory_usage(deep=True).sum()))

    return UploadResponse(
        session_id=session.session_id, filename=filename,
        n_rows=df.shape[0], n_cols=df.shape[1], columns=[str(c) for c in df.columns],
    )


# --------------------------------------------------------------------------
# Step metadata
# --------------------------------------------------------------------------
@app.get("/api/config")
def config():
    """Deployment-wide default (from OPENAI_API_KEY / .env), no session involved."""
    return {"llm_available": llm_helper.is_available(), "llm_model": llm_helper.LLM_MODEL}


@app.post("/api/eda/{session_id}/llm/connect", response_model=ApiKeyResponse)
async def connect_llm(session_id: str, body: ApiKeyRequest):
    """
    Accepts an API key typed into the UI at runtime, validates it with one
    real (cheap) call to the provider, and if valid stores it in memory on
    this session only -- never written to disk, .env, or any log line.
    """
    session = _get_session_or_404(session_id)

    now = time.time()
    last_attempt = _last_key_attempt.get(session_id, 0.0)
    if now - last_attempt < KEY_ATTEMPT_COOLDOWN_SEC:
        raise HTTPException(429, "Please wait a few seconds before trying another key.")
    _last_key_attempt[session_id] = now

    valid, message = await llm_helper.validate_api_key(body.api_key)
    if valid:
        session.llm_api_key = body.api_key.strip()
    return ApiKeyResponse(valid=valid, message=message,
                           llm_available=llm_helper.is_available(session.llm_api_key))


@app.post("/api/eda/{session_id}/llm/disconnect")
def disconnect_llm(session_id: str):
    session = _get_session_or_404(session_id)
    session.llm_api_key = None
    return {"session_id": session_id, "llm_available": llm_helper.is_available(session.llm_api_key)}


@app.get("/api/eda/{session_id}/llm/status", response_model=LlmStatusResponse)
def llm_status(session_id: str):
    session = _get_session_or_404(session_id)
    source = "session" if session.llm_api_key else ("env" if llm_helper.ENV_API_KEY else "none")
    return LlmStatusResponse(
        llm_available=llm_helper.is_available(session.llm_api_key),
        source=source, llm_model=llm_helper.LLM_MODEL,
    )


@app.post("/api/eda/{session_id}/business-context")
def set_business_context(session_id: str, body: BusinessContextRequest):
    session = _get_session_or_404(session_id)
    session.business_context = body.business_context.strip() or None
    return {"session_id": session_id, "business_context": session.business_context}


@app.get("/api/steps")
def list_steps():
    descriptions = {
        "preprocessing": "Data Loading & Preprocessing",
        "structure": "Understanding Data Structure",
        "missing_values": "Detecting Missing Values",
        "outliers": "Identifying Outliers",
        "correlation": "Finding Patterns & Correlations",
        "distributions": "Visualizing Distributions",
        "assumptions": "Checking Assumptions",
        "dimensionality": "Dimensionality Reduction (PCA)",
        "multivariate": "Multivariate Analysis",
        "summary": "Summary & Insights",
    }
    return [{"step": i + 1, "key": key, "label": descriptions[key]} for i, key in enumerate(STEP_ORDER)]


# --------------------------------------------------------------------------
# Run a single agent / the full pipeline
# --------------------------------------------------------------------------
@app.post("/api/eda/{session_id}/run/{step}", response_model=StepResult)
async def run_step(session_id: str, step: str):
    session = _get_session_or_404(session_id)
    if step not in AGENT_REGISTRY:
        raise HTTPException(404, f"Unknown step '{step}'. Valid steps: {STEP_ORDER}")
    result = await run_single_step(session, step)
    return result


@app.post("/api/eda/{session_id}/run-all", response_model=PipelineStatus)
def run_all(session_id: str, background_tasks: BackgroundTasks):
    session = _get_session_or_404(session_id)
    session.cancelled = False
    background_tasks.add_task(run_full_pipeline, session)
    session.status = "running"
    return PipelineStatus(session_id=session_id, status="running", completed_steps=[])


@app.post("/api/eda/{session_id}/cancel")
def cancel(session_id: str):
    session = _get_session_or_404(session_id)
    session.cancelled = True
    return {"session_id": session_id, "status": "cancellation_requested"}


@app.get("/api/eda/{session_id}/status", response_model=PipelineStatus)
def status(session_id: str):
    session = _get_session_or_404(session_id)
    return PipelineStatus(session_id=session_id, status=session.status,
                           completed_steps=list(session.results.keys()))


@app.get("/api/eda/{session_id}/results")
def results(session_id: str):
    session = _get_session_or_404(session_id)
    return session.results


@app.get("/api/eda/{session_id}/results/{step}", response_model=StepResult)
def step_result(session_id: str, step: str):
    session = _get_session_or_404(session_id)
    if step not in session.results:
        raise HTTPException(404, f"Step '{step}' has not been run yet for this session")
    return session.results[step]


@app.post("/api/eda/{session_id}/annotate/{step}", response_model=StepResult)
async def annotate_step(session_id: str, step: str):
    """
    Re-run only the LLM commentary for a step that already has stats computed
    -- does not recompute pandas/scipy/sklearn results. Use this after
    connecting an API key or saving business context *after* the pipeline
    already ran, so old results pick up the new key/context without a full
    re-run.
    """
    session = _get_session_or_404(session_id)
    if step not in AGENT_REGISTRY:
        raise HTTPException(404, f"Unknown step '{step}'. Valid steps: {STEP_ORDER}")
    existing = session.results.get(step)
    if not existing or existing.get("status") != "success":
        raise HTTPException(409, f"Step '{step}' must be run successfully before it can be annotated")
    if not llm_helper.is_available(session.llm_api_key):
        raise HTTPException(409, "No LLM key available for this session (connect one first)")

    agent = AGENT_REGISTRY[step]()
    data = existing["data"]
    if step == "summary":
        full_context = {k: v for k, v in session.results.items()}
        full_context["session_id"] = session.session_id
        report = await agent.annotate(
            data, session.session_id, full_context,
            api_key=session.llm_api_key, business_context=session.business_context,
        )
        if report:
            data["business_case_report"] = report
    else:
        note = await agent.annotate(data, session.session_id, api_key=session.llm_api_key)
        if note:
            data["ai_commentary"] = note

    return existing


@app.get("/api/eda/{session_id}/report")
def report(session_id: str):
    session = _get_session_or_404(session_id)
    summary = session.results.get("summary")
    if summary is None:
        raise HTTPException(409, "Run the 'summary' step (or run-all) before requesting the report")
    return {
        "session_id": session_id,
        "filename": session.filename,
        "status": session.status,
        "insights": summary.get("data", {}).get("insights", []),
        "steps": {k: v.get("status") for k, v in session.results.items()},
    }


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------
@app.get("/api/logs/{session_id}")
def logs(session_id: str):
    return get_events(session_id)


def _get_session_or_404(session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404, "Unknown session_id. Upload a dataset first via /api/upload")
    return session


# --------------------------------------------------------------------------
# Frontend (static files)
# --------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))