"""
WorkflowState — the single object every agent reads from and writes to.

LangGraph passes this dict from node to node. Each agent adds its own keys
and leaves the rest untouched, so by the time the Reporter runs it can see
everything the earlier agents produced.
"""

from __future__ import annotations

from typing import Any, TypedDict

import pandas as pd


class AgentLog(TypedDict):
    """One row of the execution trace, written by every agent."""

    agent: str
    duration_ms: int
    status: str  # "success" | "failed" | "skipped"
    note: str


class WorkflowState(TypedDict, total=False):
    """State shared across the whole agent pipeline.

    total=False means every key is optional — agents fill them in as they run.
    """

    # --- Set at workflow start -------------------------------------------
    analysis_id: str
    filename: str
    raw_df: pd.DataFrame

    # --- Written by PlannerAgent -----------------------------------------
    profile: dict[str, Any]
    plan: dict[str, Any]

    # --- Written by CleanerAgent -----------------------------------------
    clean_df: pd.DataFrame
    transformations: list[dict[str, Any]]

    # --- Written by AnalyzerAgent ----------------------------------------
    results: dict[str, Any]

    # --- Written by VisualizerAgent --------------------------------------
    charts: list[dict[str, Any]]

    # --- Written by ReporterAgent ----------------------------------------
    report: dict[str, Any]

    # --- Written by the cost ledger ---------------------------------------
    cost: dict[str, Any]

    # --- Bookkeeping ------------------------------------------------------
    status: str  # "running" | "completed" | "failed"
    current_agent: str
    error: str | None
    log: list[AgentLog]


def new_state(analysis_id: str, filename: str, df: pd.DataFrame) -> WorkflowState:
    """Build a fresh state object at the start of a run."""
    return WorkflowState(
        analysis_id=analysis_id,
        filename=filename,
        raw_df=df,
        status="running",
        current_agent="",
        error=None,
        log=[],
        transformations=[],
        charts=[],
        results={},
    )
