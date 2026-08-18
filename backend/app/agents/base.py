"""
BaseAgent — timing, logging, and error containment for every agent.

Subclasses implement `run()`. The wrapper handles the parts that would
otherwise be copy-pasted five times: measuring duration, appending to the
execution log, and making sure one agent's exception doesn't take down
the whole pipeline silently.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

from app.utils.llm import LLMClient
from app.workflow.state import WorkflowState

log = logging.getLogger(__name__)


class BaseAgent(ABC):
    #: Short identifier used in logs and progress events.
    name: str = "agent"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()
        self.note: str = ""

    @abstractmethod
    def run(self, state: WorkflowState) -> WorkflowState:
        """Do the work. Mutate and return the state."""

    def __call__(self, state: WorkflowState) -> WorkflowState:
        """LangGraph calls the agent as a plain function."""
        start = time.perf_counter()
        state["current_agent"] = self.name
        self.note = ""

        # So the ledger can attribute spend to the agent that caused it
        self.llm.agent = self.name

        try:
            log.info("[%s] starting", self.name)
            state = self.run(state)
            status = "success"
        except Exception as exc:
            log.exception("[%s] failed", self.name)
            state["status"] = "failed"
            state["error"] = f"{self.name}: {exc}"
            status = "failed"
            self.note = str(exc)[:200]

        duration_ms = int((time.perf_counter() - start) * 1000)
        state.setdefault("log", []).append(
            {
                "agent": self.name,
                "duration_ms": duration_ms,
                "status": status,
                "note": self.note,
            }
        )
        log.info("[%s] %s in %dms — %s", self.name, status, duration_ms, self.note)
        return state
