"""
HTTP and WebSocket endpoints.

Routes stay thin — parse, delegate to the service layer, shape the response.
Progress is broadcast over a WebSocket keyed by analysis ID; the in-memory
broker below is fine for one process and would become Redis pub/sub if you
ever run more than one.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from app.api.service import create_analysis, run_analysis
from app.data.ingestion import IngestionError, read_bytes
from app.db.models import Analysis, get_session

log = logging.getLogger(__name__)
router = APIRouter()


class ProgressBroker:
    """Fan out pipeline progress to any WebSocket listening for an analysis."""

    def __init__(self) -> None:
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, analysis_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[analysis_id].append(queue)
        return queue

    def unsubscribe(self, analysis_id: str, queue: asyncio.Queue) -> None:
        if queue in self._queues.get(analysis_id, []):
            self._queues[analysis_id].remove(queue)
        if not self._queues.get(analysis_id):
            self._queues.pop(analysis_id, None)

    def publish(self, analysis_id: str, event: dict) -> None:
        """Called from the background worker thread, so hop back to the loop."""
        if not self._loop:
            return
        for queue in self._queues.get(analysis_id, []):
            self._loop.call_soon_threadsafe(queue.put_nowait, event)


broker = ProgressBroker()

STEP_LABELS = {
    "planner": "Deciding which analyses to run",
    "cleaner": "Cleaning the data",
    "analyzer": "Running statistics",
    "visualizer": "Building charts",
    "reporter": "Writing the report",
}


@router.post("/analyses", status_code=202)
async def upload(
    background: BackgroundTasks,
    file: UploadFile = File(...),
    session: Session = Depends(get_session),
):
    """Accept a spreadsheet and start the analysis in the background."""
    content = await file.read()

    try:
        df = read_bytes(content, file.filename or "upload.csv")
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    analysis = create_analysis(session, file.filename or "upload.csv", df)
    analysis_id = analysis.id

    def on_progress(agent: str, percent: int) -> None:
        broker.publish(
            analysis_id,
            {
                "event": "progress",
                "agent": agent,
                "label": STEP_LABELS.get(agent, agent),
                "percent": percent,
            },
        )
        if percent >= 100:
            broker.publish(analysis_id, {"event": "done", "analysis_id": analysis_id})

    background.add_task(run_analysis, analysis_id, analysis.filename, df, on_progress)

    return {
        "analysis_id": analysis_id,
        "status": "running",
        "rows": len(df),
        "columns": len(df.columns),
    }


@router.get("/analyses")
def list_analyses(limit: int = 20, session: Session = Depends(get_session)):
    """Most recent analyses first."""
    rows = session.query(Analysis).order_by(Analysis.created_at.desc()).limit(min(limit, 100)).all()
    return {"analyses": [a.to_summary() for a in rows]}


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: str, session: Session = Depends(get_session)):
    """Everything about one analysis: profile, results, charts, report."""
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="No analysis with that ID.")
    return analysis.to_detail()


@router.get("/analyses/{analysis_id}/report", response_class=PlainTextResponse)
def get_report(analysis_id: str, session: Session = Depends(get_session)):
    """The report as raw markdown, ready to save or paste elsewhere."""
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="No analysis with that ID.")
    if analysis.report is None:
        raise HTTPException(
            status_code=409,
            detail=f"The report isn't ready — this analysis is {analysis.status}.",
        )
    return analysis.report.markdown


@router.delete("/analyses/{analysis_id}", status_code=204)
def delete_analysis(analysis_id: str, session: Session = Depends(get_session)):
    analysis = session.get(Analysis, analysis_id)
    if analysis is None:
        raise HTTPException(status_code=404, detail="No analysis with that ID.")
    session.delete(analysis)
    session.commit()


@router.websocket("/ws/{analysis_id}")
async def progress_socket(websocket: WebSocket, analysis_id: str):
    """Stream progress events while the pipeline runs."""
    await websocket.accept()
    broker.bind_loop(asyncio.get_running_loop())
    queue = broker.subscribe(analysis_id)

    try:
        while True:
            event = await asyncio.wait_for(queue.get(), timeout=300)
            await websocket.send_json(event)
            if event.get("event") in {"done", "error"}:
                break
    except TimeoutError:
        await websocket.send_json({"event": "error", "message": "Timed out waiting for progress."})
    except WebSocketDisconnect:
        log.info("Client disconnected from %s", analysis_id)
    finally:
        broker.unsubscribe(analysis_id, queue)
