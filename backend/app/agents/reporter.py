"""
ReporterAgent — turns numbers into a report someone will actually read.

The facts are assembled in Python first, then handed to the LLM as a compact
brief. The model writes prose; it never invents statistics, because every
number it can see was computed upstream.

If the LLM is offline, the deterministic sections still render — you lose the
narrative, not the report.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from app.agents.base import BaseAgent
from app.workflow.state import WorkflowState

log = logging.getLogger(__name__)

#: A causal claim is a stronger statement than what a correlation supports.
#: The prompt already says not to make one; this catches it if the model
#: does anyway, since an instruction in a prompt is a request, not a guarantee.
CAUSAL_LANGUAGE = re.compile(
    r"\b(causes?|caused by|leads? to|results? in|driven by|due to|because of)\b",
    re.IGNORECASE,
)

#: Numbers worth checking are ones that read as a statistic — a decimal or a
#: percentage. Plain integers ("the two caveats below") are prose, not data,
#: and checking them produces far more false positives than real catches.
NUMERIC_CLAIM = re.compile(r"-?\d[\d,]*\.\d+%?|-?\d[\d,]*%")


class ReporterAgent(BaseAgent):
    name = "reporter"

    def run(self, state: WorkflowState) -> WorkflowState:
        brief, grounded_values = self._build_brief(state)
        narrative = self._write_narrative(brief)
        grounding = self._check_grounding(narrative, grounded_values)
        findings = self._extract_findings(state)

        markdown = self._assemble(state, narrative, findings, grounding)

        state["report"] = {
            "markdown": markdown,
            "narrative": narrative,
            "key_findings": findings,
            "grounding_check": grounding,
            "generated_at": datetime.now(UTC).isoformat(),
        }
        state["status"] = "completed"
        self.note = f"{len(findings)} key findings"
        if grounding["ungrounded_numbers"] or grounding["causal_language"]:
            self.note += " (grounding check flagged items — see report)"
        return state

    # ------------------------------------------------------------------
    def _build_brief(self, state: WorkflowState) -> tuple[str, set[float]]:
        """Compact factual summary — this is all the model gets to see.

        Also returns every number embedded in it. That set is the ground
        truth the narrative gets checked against afterward — not a separate
        guess at what's "correct", just the exact values already handed to
        the model, so a mismatch means the model wrote a number it wasn't
        given, not that our own tracking disagrees with itself.
        """
        profile = state.get("profile", {})
        results = state.get("results", {})
        values: set[float] = set()

        def note(*nums: float | int | None) -> None:
            for n in nums:
                if n is not None:
                    values.add(float(n))

        lines = [
            f"Dataset: {state.get('filename', 'uploaded file')}",
            f"Shape: {profile.get('rows')} rows x {profile.get('columns')} columns",
            f"Numeric columns: {profile.get('numeric_columns', [])}",
            f"Categorical columns: {profile.get('categorical_columns', [])}",
        ]

        for change in state.get("transformations", []):
            lines.append(f"Cleaning: {change.get('detail')}")

        descriptive = results.get("descriptive_stats", {}).get("numeric", {})
        for col, stats_dict in list(descriptive.items())[:8]:
            lines.append(
                f"{col}: mean={stats_dict['mean']}, median={stats_dict['median']}, "
                f"std={stats_dict['std']}, skew={stats_dict['skew']}"
            )
            note(stats_dict["mean"], stats_dict["median"], stats_dict["std"], stats_dict["skew"])

        for pair in results.get("correlation", {}).get("notable_pairs", [])[:5]:
            lines.append(
                f"Correlation: {pair['column_a']} and {pair['column_b']} "
                f"r={pair['r']}, p={pair['p_value']}, "
                f"significant={pair['significant']}"
            )
            note(pair["r"], pair["p_value"])

        for test in results.get("group_comparison", {}).get("comparisons", [])[:5]:
            lines.append(
                f"Group test: {test['value_column']} across {test['group_column']} "
                f"({test['test']}) p={test['p_value']}, significant={test['significant']}"
            )
            note(test["p_value"])

        anomalies = results.get("outlier_detection", {})
        if anomalies.get("count"):
            lines.append(
                f"Anomalies: {anomalies['count']} rows "
                f"({anomalies['percentage']}%) flagged by Isolation Forest"
            )
            note(anomalies["count"], anomalies["percentage"])

        regression = results.get("regression", {})
        if regression.get("r2_score") is not None:
            top = regression["feature_importance"][0]["feature"]
            lines.append(
                f"Model: predicting {regression['target']}, "
                f"R2={regression['r2_score']}, strongest predictor={top}"
            )
            note(regression["r2_score"])

        # Row/column counts are legitimate things to restate in prose
        note(profile.get("rows"), profile.get("columns"))

        return "\n".join(lines), values

    def _write_narrative(self, brief: str) -> str:
        prompt = f"""Here are the results of a data analysis:

{brief}

Write a short analysis report in markdown with these sections:
## Summary — two or three sentences on what this dataset contains and what stands out.
## What the data shows — the most important patterns, with the actual numbers.
## Caveats — what these results do not prove, including any data quality issues.
## Suggested next steps — two or three concrete follow-up analyses.

Rules:
- Use only the numbers given above. Do not invent statistics.
- Correlation is not causation; say so where it matters.
- Write plainly. No filler, no restating the prompt.
"""
        try:
            return self.llm.ask(
                prompt, system="You are a careful data analyst writing for a non-specialist."
            )
        except Exception as exc:
            log.warning("Narrative generation failed: %s", exc)
            return "_Narrative unavailable — the language model could not be reached._"

    def _check_grounding(self, narrative: str, grounded_values: set[float]) -> dict:
        """Best-effort check that numbers in the narrative trace back to the brief.

        Not a hard gate — a false positive here (a real number that legitimately
        got rounded or converted to a percentage differently than expected)
        shouldn't block the report, so this flags for a human to glance at
        rather than rewriting or rejecting anything. What it catches: a number
        that appears nowhere in what the model was actually given, which is
        the specific failure mode "use only the numbers given above" exists
        to prevent.

        Deliberately loose on tolerance (rounding, sign, fraction-vs-percent
        conversion) to avoid crying wolf on correct paraphrasing — it would
        rather miss a subtle fabrication than flag normal rounding on every
        report.
        """
        candidates: set[float] = set()
        for v in grounded_values:
            for form in (v, abs(v), v * 100, abs(v) * 100, v / 100):
                for precision in (0, 1, 2, 3):
                    candidates.add(round(form, precision))

        def is_grounded(x: float) -> bool:
            return any(abs(x - c) <= max(0.05, 0.01 * abs(c)) for c in candidates)

        ungrounded = []
        for match in NUMERIC_CLAIM.finditer(narrative):
            token = match.group().replace(",", "")
            try:
                value = float(token.rstrip("%"))
            except ValueError:
                continue
            if not is_grounded(value):
                ungrounded.append(match.group())

        causal_hits = sorted({m.group() for m in CAUSAL_LANGUAGE.finditer(narrative)})

        return {
            "ungrounded_numbers": sorted(set(ungrounded)),
            "causal_language": causal_hits,
        }

    def _extract_findings(self, state: WorkflowState) -> list[str]:
        """Deterministic bullet points, derived straight from the numbers."""
        results = state.get("results", {})
        findings: list[str] = []

        for pair in results.get("correlation", {}).get("notable_pairs", [])[:3]:
            if pair["significant"]:
                findings.append(
                    f"{pair['column_a']} and {pair['column_b']} move together "
                    f"({pair['direction']}, r = {pair['r']:.2f}, p = {pair['p_value']:.3g})."
                )

        for test in results.get("group_comparison", {}).get("comparisons", [])[:3]:
            if test["significant"]:
                findings.append(
                    f"{test['value_column']} differs significantly across "
                    f"{test['group_column']} groups ({test['test']}, p = {test['p_value']:.3g})."
                )

        anomalies = results.get("outlier_detection", {})
        if anomalies.get("count"):
            findings.append(
                f"{anomalies['count']} rows ({anomalies['percentage']:.1f}%) "
                f"look anomalous compared with the rest of the data."
            )

        regression = results.get("regression", {})
        if regression.get("r2_score") is not None:
            top = regression["feature_importance"][0]
            findings.append(
                f"A model of {regression['target']} explains "
                f"{regression['r2_score'] * 100:.0f}% of its variance, "
                f"led by {top['feature']}."
            )

        non_normal = [
            t["column"]
            for t in results.get("distribution_tests", {}).get("normality", [])
            if not t["normally_distributed"]
        ]
        if non_normal:
            findings.append(
                f"{len(non_normal)} column(s) are not normally distributed "
                f"({', '.join(non_normal[:3])}), so rank-based tests may fit better."
            )

        return findings or ["No statistically significant patterns surfaced in this dataset."]

    def _assemble(
        self, state: WorkflowState, narrative: str, findings: list[str], grounding: dict
    ) -> str:
        profile = state.get("profile", {})
        parts = [
            f"# Analysis: {state.get('filename', 'dataset')}",
            "",
            f"*{profile.get('rows', 0):,} rows · {profile.get('columns', 0)} columns · "
            f"generated {datetime.now(UTC):%d %b %Y %H:%M UTC}*",
            "",
            narrative,
            "",
            "## Key findings",
            "",
        ]
        parts.extend(f"- {f}" for f in findings)

        # Silent when clean — this is a QA signal for the rare case, not a
        # section every report carries. Not a claim any flagged item is
        # wrong, just that it couldn't be traced back to a computed value.
        if grounding["ungrounded_numbers"] or grounding["causal_language"]:
            parts += ["", "## Narrative quality check", ""]
            if grounding["ungrounded_numbers"]:
                nums = ", ".join(grounding["ungrounded_numbers"])
                parts.append(f"- Numbers in the narrative not traced to a computed value: {nums}")
            if grounding["causal_language"]:
                phrases = ", ".join(f'"{p}"' for p in grounding["causal_language"])
                parts.append(f"- Causal language used despite the correlation-only data: {phrases}")

        changes = state.get("transformations", [])
        if changes:
            parts += ["", "## What was changed before analysis", ""]
            parts.extend(f"- {c.get('detail')}" for c in changes)

        charts = state.get("charts", [])
        if charts:
            parts += ["", "## Charts", ""]
            parts.extend(f"- **{c['title']}** — {c['description']}" for c in charts)

        parts += ["", "## Run log", "", "| Agent | Status | Time |", "|---|---|---|"]
        for entry in state.get("log", []):
            parts.append(f"| {entry['agent']} | {entry['status']} | {entry['duration_ms']} ms |")

        return "\n".join(parts)
