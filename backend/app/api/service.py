"""
The service layer between HTTP and the agents.

Keeps the route handlers thin: they parse the request and hand off here.
This is also where pipeline output gets written to the database, so the
agents never need to know persistence exists.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import Analysis, Chart, Report, SessionLocal
from app.workflow.pipeline import run_pipeline
from app.workflow.state import new_state

log = logging.getLogger(__name__)


def create_analysis(session: Session, filename: str, df: pd.DataFrame) -> Analysis:
    """Record the upload before any work starts, so the ID exists immediately."""
    analysis = Analysis(
        filename=filename,
        status="running",
        row_count=len(df),
        column_count=len(df.columns),
    )
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
    log.info("Created analysis %s for %s", analysis.id, filename)
    return analysis


def run_analysis(
    analysis_id: str,
    filename: str,
    df: pd.DataFrame,
    on_progress: Callable[[str, int], None] | None = None,
) -> None:
    """Run the pipeline and save the result.

    Opens its own session because this runs in a background task, outside the
    request's session lifecycle.
    """
    session = SessionLocal()
    try:
        state = new_state(analysis_id, filename, df)
        final = run_pipeline(state, on_progress=on_progress)
        _persist(session, analysis_id, final)
    except Exception as exc:
        log.exception("Analysis %s failed", analysis_id)
        _mark_failed(session, analysis_id, str(exc))
    finally:
        session.close()


def _persist(session: Session, analysis_id: str, state: dict) -> None:
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        log.error("Analysis %s vanished before results could be saved", analysis_id)
        return

    analysis.status = state.get("status", "completed")
    analysis.error = state.get("error")
    analysis.profile = state.get("profile")
    analysis.plan = state.get("plan")
    analysis.transformations = state.get("transformations")
    analysis.results = state.get("results")
    analysis.run_log = state.get("log")
    analysis.completed_at = datetime.now(UTC)

    clean_df = state.get("clean_df")
    if clean_df is not None:
        analysis.row_count = len(clean_df)
        analysis.column_count = len(clean_df.columns)

    for chart in state.get("charts", []):
        session.add(
            Chart(
                analysis_id=analysis_id,
                chart_type=chart["chart_type"],
                title=chart["title"],
                description=chart.get("description"),
                plotly_json=chart["plotly_json"],
            )
        )

    report = state.get("report")
    if report:
        session.add(
            Report(
                analysis_id=analysis_id,
                markdown=report["markdown"],
                narrative=report.get("narrative"),
                key_findings=report.get("key_findings"),
            )
        )

    session.commit()
    log.info("Saved analysis %s (%s)", analysis_id, analysis.status)


def _mark_failed(session: Session, analysis_id: str, error: str) -> None:
    analysis = session.get(Analysis, analysis_id)
    if analysis:
        analysis.status = "failed"
        analysis.error = error[:1000]
        analysis.completed_at = datetime.now(UTC)
        session.commit()
