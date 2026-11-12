"""
Counting what each run actually costs.

Every model call records its token usage and price so the run log can show
where the money went, the same way the timing strip shows where the time went.

Prices are per million tokens and change; they live here as a single table
rather than scattered through the code so updating them is one edit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: USD per million tokens, as (input, output). Update when providers change them.
PRICES: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    # Anthropic
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4": (15.00, 75.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Used when a model isn't in the table, so cost is an estimate rather than zero.
FALLBACK_PRICE = (2.50, 10.00)


@dataclass
class CallRecord:
    """One request to a model provider."""

    agent: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached: bool = False

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class CostLedger:
    """Running total for a single analysis.

    Shared by reference across every agent, so the whole run accumulates into
    one object that the reporter and HTML export can read.
    """

    calls: list[CallRecord] = field(default_factory=list)

    def record(
        self,
        agent: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached: bool = False,
    ) -> CallRecord:
        entry = CallRecord(
            agent=agent,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=price_call(model, input_tokens, output_tokens),
            cached=cached,
        )
        self.calls.append(entry)
        log.info(
            "[%s] %s — %d in, %d out, $%.5f%s",
            agent,
            model,
            input_tokens,
            output_tokens,
            entry.cost_usd,
            " (cached)" if cached else "",
        )
        return entry

    def record_cache_hit(self, agent: str, model: str) -> CallRecord:
        """A call that was avoided entirely. Zero tokens, zero cost."""
        return self.record(agent, model, 0, 0, cached=True)

    @property
    def total_cost(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def billed_calls(self) -> int:
        """Calls that actually hit the provider."""
        return sum(1 for c in self.calls if not c.cached)

    def summary(self) -> dict:
        """Serialisable view for the run log and the HTML report."""
        by_agent: dict[str, dict] = {}
        for call in self.calls:
            bucket = by_agent.setdefault(
                call.agent,
                {"calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0},
            )
            bucket["calls"] += 1
            bucket["cached"] += int(call.cached)
            bucket["input_tokens"] += call.input_tokens
            bucket["output_tokens"] += call.output_tokens
            bucket["cost_usd"] += call.cost_usd

        return {
            "total_cost_usd": round(self.total_cost, 6),
            "total_tokens": self.total_tokens,
            "total_calls": len(self.calls),
            "billed_calls": self.billed_calls,
            "by_agent": {
                agent: {**stats, "cost_usd": round(stats["cost_usd"], 6)}
                for agent, stats in by_agent.items()
            },
        }


def price_call(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for one call.

    Model names carry date suffixes (gpt-4o-2024-08-06), so match on the
    longest known prefix rather than requiring an exact key.
    """
    price_in, price_out = _lookup(model)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def _lookup(model: str) -> tuple[float, float]:
    if model in PRICES:
        return PRICES[model]

    # Longest match first, so gpt-4o-mini doesn't resolve to gpt-4o
    for known in sorted(PRICES, key=len, reverse=True):
        if model.startswith(known):
            return PRICES[known]

    log.debug("No price for %s; using fallback", model)
    return FALLBACK_PRICE
