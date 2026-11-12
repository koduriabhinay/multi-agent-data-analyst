"""Web interface API — file upload and analysis."""

from pathlib import Path
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
import logging

from app.data.ingestion import read_bytes
from app.workflow.pipeline import run_pipeline_sync
from app.workflow.state import new_state

log = logging.getLogger(__name__)

def create_web_app() -> FastAPI:
    app = FastAPI(title="MADA", description="Multi-Agent Data Analyst")

    @app.post("/api/analyze")
    async def analyze(file: UploadFile):
        """Upload a file, run analysis, return results."""
        try:
            content = await file.read()
            df = read_bytes(content, file.filename)
            state = new_state("web-upload", file.filename, df)
            result = run_pipeline_sync(state)

            return {
                "profile": result.get("profile", {}),
                "report": result.get("report", {}),
                "charts": result.get("charts", []),
                "cost": result.get("cost", {}),
                "transformations": result.get("transformations", []),
            }
        except Exception as e:
            log.error("Analysis failed: %s", e)
            raise HTTPException(status_code=400, detail=str(e))

    # Serve static UI
    static_dir = Path(__file__).parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app
