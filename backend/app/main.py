"""
Application entry point.

    uvicorn app.main:app --reload

Docs are at /docs, the API is under /api.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

from app.api.routes import router  # noqa: E402 — must follow load_dotenv
from app.db.models import init_db  # noqa: E402
from app.utils.llm import LLMClient  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-7s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mada")

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("Database ready")

    probe = LLMClient()
    if probe.offline:
        log.warning(
            "No API key found — running in offline mode. "
            "Statistics and charts work; the written narrative will be stubbed. "
            "Set ANTHROPIC_API_KEY in .env to enable it."
        )
    else:
        log.info("Language model ready: %s / %s", probe.provider, probe.model)

    yield
    log.info("Shutting down")


app = FastAPI(
    title="Multi-Agent Data Analyst",
    description=(
        "Upload a spreadsheet. Five agents profile it, clean it, run statistics, "
        "build charts, and write up what they found."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/")
def index():
    return {
        "name": "Multi-Agent Data Analyst",
        "docs": "/docs",
        "health": "/health",
        "upload": "POST /api/analyses",
    }


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception):
    """Log the detail, return something the user can act on."""
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong on our side. The error has been logged."},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
