"""
Persistence layer.

Defaults to SQLite so the project runs with zero setup. Point DATABASE_URL at
Postgres and the same models work unchanged — JSON columns are portable across
both, which is why results are stored as JSON rather than wide tables.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./analyst.db")

# check_same_thread is a SQLite-only quirk: FastAPI touches sessions across threads
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=os.getenv("SQL_ECHO", "").lower() == "true",
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Analysis(Base):
    """One uploaded file and everything derived from it."""

    __tablename__ = "analyses"

    id = Column(String(36), primary_key=True, default=_uuid)
    filename = Column(String(255), nullable=False)
    status = Column(String(20), nullable=False, default="running", index=True)
    error = Column(Text, nullable=True)

    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)

    profile = Column(JSON, nullable=True)
    plan = Column(JSON, nullable=True)
    transformations = Column(JSON, nullable=True)
    results = Column(JSON, nullable=True)
    run_log = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=_now, nullable=False)
    completed_at = Column(DateTime, nullable=True)

    charts = relationship("Chart", back_populates="analysis", cascade="all, delete-orphan")
    report = relationship(
        "Report",
        back_populates="analysis",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def to_summary(self) -> dict:
        """Lightweight view for list endpoints — no chart payloads."""
        return {
            "id": self.id,
            "filename": self.filename,
            "status": self.status,
            "error": self.error,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def to_detail(self) -> dict:
        """Full view, including charts and the report."""
        return {
            **self.to_summary(),
            "profile": self.profile,
            "plan": self.plan,
            "transformations": self.transformations,
            "results": self.results,
            "run_log": self.run_log,
            "charts": [c.to_dict() for c in self.charts],
            "report": self.report.to_dict() if self.report else None,
        }


class Chart(Base):
    __tablename__ = "charts"

    id = Column(String(36), primary_key=True, default=_uuid)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    chart_type = Column(String(40), nullable=False)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    plotly_json = Column(JSON, nullable=False)

    analysis = relationship("Analysis", back_populates="charts")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "chart_type": self.chart_type,
            "title": self.title,
            "description": self.description,
            "plotly_json": self.plotly_json,
        }


class Report(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=_uuid)
    analysis_id = Column(String(36), ForeignKey("analyses.id", ondelete="CASCADE"), index=True)
    markdown = Column(Text, nullable=False)
    narrative = Column(Text, nullable=True)
    key_findings = Column(JSON, nullable=True)
    generated_at = Column(DateTime, default=_now)

    analysis = relationship("Analysis", back_populates="report")

    def to_dict(self) -> dict:
        return {
            "markdown": self.markdown,
            "narrative": self.narrative,
            "key_findings": self.key_findings,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


def init_db() -> None:
    """Create tables. Safe to call repeatedly."""
    Base.metadata.create_all(bind=engine)


def get_session() -> Session:
    """FastAPI dependency — yields a session and always closes it."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
