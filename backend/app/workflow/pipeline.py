"""
The pipeline — five agents, wired in sequence by LangGraph.

Each node is an agent instance. LangGraph handles passing state between them
and gives us streaming for free, which is what feeds the progress WebSocket.

There's a plain-Python fallback (`run_pipeline_sync`) so the analysis still
works if LangGraph isn't installed. Useful for tests and for anyone who just
wants to run the CLI.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from itertools import pairwise
from typing import Any

from app.agents import (
    AnalyzerAgent,
    CleanerAgent,
    PlannerAgent,
    ReporterAgent,
    VisualizerAgent,
)
from app.utils.cost import CostLedger
from app.utils.llm import LLMClient
from app.workflow.state import WorkflowState

log = logging.getLogger(__name__)

#: Order matters — each agent depends on what the previous one wrote.
AGENT_SEQUENCE = ["planner", "cleaner", "analyzer", "visualizer", "reporter"]

#: Rough progress percentage after each agent finishes, for the UI.
PROGRESS = {
    "planner": 15,
    "cleaner": 35,
    "analyzer": 60,
    "visualizer": 80,
    "reporter": 100,
}


def build_agents(llm: LLMClient | None = None) -> dict[str, Any]:
    """One shared LLM client across all agents.

    Sharing matters for more than setup cost: the client owns the prompt cache
    and the cost ledger, so every agent contributes to the same totals.
    """
    llm = llm or LLMClient(ledger=CostLedger())
    return {
        "planner": PlannerAgent(llm),
        "cleaner": CleanerAgent(llm),
        "analyzer": AnalyzerAgent(llm),
        "visualizer": VisualizerAgent(llm),
        "reporter": ReporterAgent(llm),
    }


def build_graph(llm: LLMClient | None = None):
    """Compile the LangGraph state machine."""
    from langgraph.graph import END, StateGraph

    agents = build_agents(llm)
    graph = StateGraph(WorkflowState)

    for name in AGENT_SEQUENCE:
        graph.add_node(name, agents[name])

    graph.set_entry_point(AGENT_SEQUENCE[0])
    for current, following in pairwise(AGENT_SEQUENCE):
        graph.add_edge(current, following)
    graph.add_edge(AGENT_SEQUENCE[-1], END)

    return graph.compile()


def run_pipeline(
    state: WorkflowState,
    llm: LLMClient | None = None,
    on_progress: Callable[[str, int], None] | None = None,
) -> WorkflowState:
    """Run the full analysis.

    `on_progress(agent_name, percent)` is called after each agent so callers
    can stream updates to a UI. Errors are captured in state rather than raised,
    so a partial result is still returned.
    """
    agents = build_agents(llm)
    shared_llm = agents["planner"].llm

    try:
        final = _run_with_langgraph(state, shared_llm, on_progress)
    except ImportError:
        log.info("LangGraph not installed — running agents directly")
        final = run_pipeline_sync(state, shared_llm, on_progress)

    final["cost"] = shared_llm.ledger.summary()
    log.info(
        "Run cost $%.5f across %d billed calls",
        shared_llm.ledger.total_cost,
        shared_llm.ledger.billed_calls,
    )
    return final


def _run_with_langgraph(
    state: WorkflowState,
    llm: LLMClient | None,
    on_progress: Callable[[str, int], None] | None,
) -> WorkflowState:
    graph = build_graph(llm)
    final = state

    for step in graph.stream(state):
        for node_name, node_state in step.items():
            final = node_state
            if on_progress:
                on_progress(node_name, PROGRESS.get(node_name, 0))
            if node_state.get("status") == "failed":
                log.error("Pipeline halted at %s: %s", node_name, node_state.get("error"))
                return node_state

    return final


def run_pipeline_sync(
    state: WorkflowState,
    llm: LLMClient | None = None,
    on_progress: Callable[[str, int], None] | None = None,
) -> WorkflowState:
    """Same pipeline without the LangGraph dependency."""
    agents = build_agents(llm)

    for name in AGENT_SEQUENCE:
        state = agents[name](state)
        if on_progress:
            on_progress(name, PROGRESS[name])
        if state.get("status") == "failed":
            log.error("Pipeline halted at %s: %s", name, state.get("error"))
            break

    return state
