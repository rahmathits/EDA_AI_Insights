from __future__ import annotations

import asyncio
from typing import Any, Dict

import llm_helper
from agents import AGENT_REGISTRY, PIPELINE
from logger_config import log_report_generation_failure, log_user_cancellation
from session_store import Session


async def run_single_step(session: Session, step_name: str) -> Dict[str, Any]:
    """
    Run one named agent against the session's current dataframe, then --
    if an LLM key is available (session runtime key or OPENAI_API_KEY) --
    ask it to interpret that agent's results.
    """
    agent_cls = AGENT_REGISTRY.get(step_name)
    if agent_cls is None:
        raise KeyError(f"Unknown EDA step: {step_name}")

    agent = agent_cls()
    context = {k: v for k, v in session.results.items()}
    context["session_id"] = session.session_id
    result = agent.execute(session.df, session.session_id, context)

    # Special case: preprocessing agent may return a cleaned dataframe.
    data = result.get("data")
    if isinstance(data, dict) and "_cleaned_df" in data:
        session.df = data.pop("_cleaned_df")

    if result.get("status") == "success" and llm_helper.is_available(session.llm_api_key):
        if step_name == "summary":
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

    session.results[step_name] = result
    return result


async def run_full_pipeline(session: Session) -> Dict[str, Any]:
    """
    Run all 10 steps in order (fast, deterministic stats), checking
    session.cancelled between each step. If an LLM key is available
    (session runtime key or OPENAI_API_KEY), the interpretive commentary
    for steps 1-9 is then generated concurrently (they're independent of
    each other), and finally the Summary agent synthesizes everything --
    including those commentaries plus any business_context the user
    provided -- into a structured 5-section business-case report.
    """
    session.status = "running"
    agent_instances: Dict[str, Any] = {}

    for agent_cls in PIPELINE:
        if session.cancelled:
            session.status = "cancelled"
            log_user_cancellation(session.session_id, stage=agent_cls().name)
            return {"status": "cancelled", "completed_steps": list(session.results.keys())}

        agent = agent_cls()
        agent_instances[agent.name] = agent
        context = {k: v for k, v in session.results.items()}
        context["session_id"] = session.session_id
        try:
            result = agent.execute(session.df, session.session_id, context)
        except Exception as exc:  # noqa: BLE001
            log_report_generation_failure(session.session_id, reason=str(exc))
            session.status = "error"
            return {"status": "error", "error": str(exc), "completed_steps": list(session.results.keys())}

        data = result.get("data")
        if isinstance(data, dict) and "_cleaned_df" in data:
            session.df = data.pop("_cleaned_df")
        session.results[agent.name] = result

    if llm_helper.is_available(session.llm_api_key):
        if session.cancelled:
            session.status = "cancelled"
            log_user_cancellation(session.session_id, stage="llm_annotation")
            return {"status": "cancelled", "completed_steps": list(session.results.keys())}

        # Steps 1-9: independent per-step commentary, fired concurrently.
        tasks, keys = [], []
        for key, result in session.results.items():
            if key == "summary" or result.get("status") != "success":
                continue
            tasks.append(agent_instances[key].annotate(
                result["data"], session.session_id, api_key=session.llm_api_key
            ))
            keys.append(key)

        if tasks:
            annotations = await asyncio.gather(*tasks, return_exceptions=True)
            for key, note in zip(keys, annotations):
                if isinstance(note, Exception) or not note:
                    continue
                session.results[key]["data"]["ai_commentary"] = note

        # Step 10: structured business-case synthesis, now that per-step
        # commentary exists and using any business_context the user set.
        summary_result = session.results.get("summary")
        if summary_result and summary_result.get("status") == "success":
            full_context = {k: v for k, v in session.results.items()}
            full_context["session_id"] = session.session_id
            report = await agent_instances["summary"].annotate(
                summary_result["data"], session.session_id, full_context,
                api_key=session.llm_api_key, business_context=session.business_context,
            )
            if report:
                summary_result["data"]["business_case_report"] = report

    session.status = "complete"
    return {"status": "complete", "completed_steps": list(session.results.keys())}