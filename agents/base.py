from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, Optional

import pandas as pd

import llm_helper
from logger_config import (
    MemoryTracker,
    log_agent_complete,
    log_agent_start,
    log_memory_usage,
    log_retry_attempt,
    log_statistical_failure,
    log_warning,
)


class AgentError(Exception):
    """Raised when an agent cannot complete even after retries."""


class BaseAgent:
    """
    Every EDA step is a small, self-contained agent. Agents share a uniform
    run() contract so the orchestrator can execute, time, retry and log them
    identically:

        result = agent.execute(df, session_id, context)

    context is a dict of results from previously-run agents, so later agents
    (e.g. Summary) can build on earlier findings without recomputation.
    """

    name: str = "base_agent"
    max_retries: int = 2
    use_llm: bool = True  # set False on a subclass to skip LLM enhancement for that step

    def run(self, df: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def execute(self, df: pd.DataFrame, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        log_agent_start(session_id, self.name)
        start = time.time()
        last_exc: Exception | None = None

        with MemoryTracker() as tracker:
            for attempt in range(1, self.max_retries + 2):
                try:
                    result = self.run(df, context)
                    duration = time.time() - start
                    current_mb, peak_mb = tracker.sample()
                    log_memory_usage(session_id, self.name, current_mb, peak_mb)
                    log_agent_complete(session_id, self.name, duration)
                    return {"agent": self.name, "status": "success", "duration_sec": round(duration, 4),
                            "data": result}
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if attempt <= self.max_retries:
                        log_retry_attempt(session_id, self.name, attempt, reason=str(exc))
                        time.sleep(0.1 * attempt)
                        continue
                    log_statistical_failure(session_id, self.name, test=self.name,
                                             reason=f"{exc}\n{traceback.format_exc(limit=2)}")

        duration = time.time() - start
        log_agent_complete(session_id, self.name, duration)
        return {
            "agent": self.name,
            "status": "failed",
            "duration_sec": round(duration, 4),
            "error": str(last_exc) if last_exc else "unknown error",
            "data": None,
        }

    @staticmethod
    def warn(session_id: str, agent: str, message: str):
        log_warning(session_id, agent, message)

    async def annotate(self, data: Dict[str, Any], session_id: str,
                        context: Optional[Dict[str, Any]] = None,
                        api_key: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Ask the LLM to interpret this agent's already-computed results.
        Returns None (no-op) if use_llm is False, no key is available
        (neither the session's runtime key nor OPENAI_API_KEY), or the call
        fails -- callers should treat that as "no commentary", never as an
        error. **kwargs absorbs extra params subclasses may add (e.g.
        SummaryAgent's business_context) so the orchestrator can call every
        agent's annotate() with a uniform signature.
        """
        if not self.use_llm or not llm_helper.is_available(api_key) or not data:
            return None
        try:
            return await llm_helper.get_commentary(self.name, data, session_id, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            self.warn(session_id, self.name, f"LLM annotate() failed: {exc}")
            return None