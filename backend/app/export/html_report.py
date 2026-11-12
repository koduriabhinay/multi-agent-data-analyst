"""
Rendering an analysis as a standalone HTML page.

The design leans on what makes this report unusual: it shows its work. Every
finding carries the statistic that produced it, the cleaning log says what was
changed before any analysis ran, and the timing strip makes visible where the
run actually spent its time. That provenance is the signature, not decoration.

Output is a single file. Plotly loads from a CDN; everything else is inline,
so the report survives being emailed around.
"""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from typing import Any

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.27.0.min.js"

#: Two signal colours, because significance is the one binary this report
#: repeats over and over. Everything else stays quiet.
STYLES = """
:root {
  --ink:      #10233A;
  --paper:    #F6F8FA;
  --surface:  #FFFFFF;
  --rule:     #DDE4EB;
  --muted:    #5A6B7B;
  --signal:   #0F6E6B;
  --flag:     #8A6A2F;
  --measure:  68ch;
}

*, *::before, *::after { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "Source Serif 4", Georgia, "Times New Roman", serif;
  font-size: 17px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

.shell { max-width: 1080px; margin: 0 auto; padding: 0 28px 96px; }

/* ---------- Masthead ---------- */
.masthead { padding: 72px 0 40px; border-bottom: 2px solid var(--ink); }

.eyebrow {
  font-family: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 0 0 14px;
}

h1 {
  font-family: "Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif;
  font-weight: 700;
  font-size: clamp(2.1rem, 5vw, 3.4rem);
  line-height: 1.04;
  letter-spacing: -0.02em;
  margin: 0 0 20px;
  overflow-wrap: anywhere;
}

.shape {
  display: flex;
  flex-wrap: wrap;
  gap: 28px;
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  font-size: 13px;
  color: var(--muted);
}
.shape b { color: var(--ink); font-weight: 600; }

/* ---------- Sections ---------- */
section { padding-top: 56px; }

h2 {
  font-family: "Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif;
  font-weight: 600;
  font-size: 1.45rem;
  letter-spacing: -0.01em;
  margin: 0 0 22px;
  padding-bottom: 10px;
  border-bottom: 1px solid var(--rule);
}

h3 {
  font-family: "Bricolage Grotesque", "Helvetica Neue", Arial, sans-serif;
  font-weight: 600;
  font-size: 1.05rem;
  margin: 32px 0 10px;
}

p { max-width: var(--measure); margin: 0 0 18px; }

/* ---------- Findings: the statistic is the headline ---------- */
.findings { display: grid; gap: 14px; }

.finding {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 3px solid var(--signal);
  border-radius: 3px;
  padding: 18px 22px;
}
.finding.is-flagged { border-left-color: var(--flag); }

.finding .stat {
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.06em;
  color: var(--signal);
  display: block;
  margin-bottom: 6px;
}
.finding.is-flagged .stat { color: var(--flag); }

.finding p { margin: 0; font-size: 16px; }

/* ---------- Charts ---------- */
.chart {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: 3px;
  margin-bottom: 28px;
  overflow: hidden;
}
.chart-caption { padding: 18px 22px 4px; }
.chart-caption h3 { margin: 0 0 4px; font-size: 1rem; }
.chart-caption p { margin: 0; font-size: 14px; color: var(--muted); max-width: var(--measure); }
.chart-plot { padding: 4px 8px 12px; min-height: 380px; }

/* ---------- Cleaning log ---------- */
.changes { list-style: none; padding: 0; margin: 0; max-width: var(--measure); }
.changes li {
  padding: 11px 0 11px 26px;
  border-bottom: 1px solid var(--rule);
  position: relative;
  font-size: 16px;
}
.changes li::before {
  content: "";
  position: absolute;
  left: 4px;
  top: 21px;
  width: 8px;
  height: 1px;
  background: var(--muted);
}
.changes li:last-child { border-bottom: none; }

/* ---------- Timing strip: the signature ---------- */
.timing { background: var(--surface); border: 1px solid var(--rule); border-radius: 3px; padding: 22px; }

.timing-bar { display: flex; height: 34px; border-radius: 2px; overflow: hidden; margin-bottom: 16px; }
.timing-bar span { display: block; min-width: 2px; }

.timing-key { display: grid; gap: 8px; }
.timing-row {
  display: grid;
  grid-template-columns: 12px 1fr auto auto;
  align-items: center;
  gap: 12px;
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  font-size: 12.5px;
}
.swatch { width: 12px; height: 12px; border-radius: 2px; }
.timing-row .ms { color: var(--muted); }
.timing-row .pct { color: var(--muted); min-width: 44px; text-align: right; }

.note {
  font-size: 13px;
  color: var(--muted);
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  margin-top: 18px;
}

footer {
  margin-top: 72px;
  padding-top: 22px;
  border-top: 1px solid var(--rule);
  font-family: "JetBrains Mono", ui-monospace, Menlo, monospace;
  font-size: 12px;
  color: var(--muted);
}

@media (max-width: 640px) {
  body { font-size: 16px; }
  .shell { padding: 0 18px 64px; }
  .masthead { padding-top: 44px; }
}

@media print {
  body { background: #fff; }
  .chart, .finding, .timing { break-inside: avoid; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
"""

#: Distinct hues for the timing strip, ordered to match the agent sequence.
AGENT_COLOURS = {
    "planner": "#0F6E6B",
    "cleaner": "#3E8C84",
    "analyzer": "#6FA8A0",
    "visualizer": "#B4864A",
    "reporter": "#10233A",
}


def render_report(state: dict[str, Any]) -> str:
    """Build the full HTML document for a finished analysis."""
    profile = state.get("profile") or {}
    report = state.get("report") or {}
    charts = state.get("charts") or []

    title = state.get("filename", "Analysis")

    body = "\n".join(
        part
        for part in (
            _masthead(title, profile, charts),
            _narrative(report.get("narrative", "")),
            _findings(report.get("key_findings", []), state.get("results", {})),
            _charts(charts),
            _changes(state.get("transformations", [])),
            _timing(state.get("log", [])),
        )
        if part
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — analysis</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400..700&family=Source+Serif+4:opsz,wght@8..60,400..600&family=JetBrains+Mono:wght@400..600&display=swap" rel="stylesheet">
<script src="{PLOTLY_CDN}" charset="utf-8"></script>
<style>{STYLES}</style>
</head>
<body>
<div class="shell">
{body}
<footer>
  Generated {datetime.now(UTC):%d %B %Y, %H:%M UTC} by Multi-Agent Data Analyst.
  Every statistic was computed with scipy and scikit-learn.
</footer>
</div>
<script>
document.querySelectorAll("[data-figure]").forEach(function (node) {{
  var figure = JSON.parse(node.getAttribute("data-figure"));
  Plotly.newPlot(node, figure.data, figure.layout, {{
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"]
  }});
}});
</script>
</body>
</html>"""


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
def _masthead(title: str, profile: dict, charts: list) -> str:
    rows = profile.get("rows", 0)
    cols = profile.get("columns", 0)
    numeric = len(profile.get("numeric_columns", []))

    return f"""<header class="masthead">
  <p class="eyebrow">Automated analysis</p>
  <h1>{html.escape(title)}</h1>
  <div class="shape">
    <span><b>{rows:,}</b> rows</span>
    <span><b>{cols}</b> columns</span>
    <span><b>{numeric}</b> numeric</span>
    <span><b>{len(charts)}</b> charts</span>
  </div>
</header>"""


def _narrative(narrative: str) -> str:
    if not narrative.strip():
        return ""
    return f"<section>{_markdown_to_html(narrative)}</section>"


def _findings(findings: list[str], results: dict) -> str:
    """Each finding leads with the statistic that produced it.

    The label is pulled from the finding text itself rather than passed in
    separately, so this stays correct if the Reporter's wording changes.
    """
    if not findings:
        return ""

    cards = []
    for finding in findings:
        label, flagged = _classify(finding)
        cards.append(
            f'<div class="finding{" is-flagged" if flagged else ""}">'
            f'<span class="stat">{html.escape(label)}</span>'
            f"<p>{html.escape(finding)}</p></div>"
        )

    return f'<section><h2>Key findings</h2><div class="findings">{"".join(cards)}</div></section>'


def _charts(charts: list[dict]) -> str:
    if not charts:
        return ""

    blocks = []
    for i, chart in enumerate(charts):
        figure = json.dumps(chart.get("plotly_json", {}))
        # Attribute-escaped so quotes in the JSON can't break out of the tag
        safe_figure = html.escape(figure, quote=True)

        blocks.append(
            f'<div class="chart">'
            f'<div class="chart-caption">'
            f"<h3>{html.escape(chart.get('title', 'Chart'))}</h3>"
            f"<p>{html.escape(chart.get('description', ''))}</p>"
            f"</div>"
            f'<div class="chart-plot" id="chart-{i}" data-figure="{safe_figure}"></div>'
            f"</div>"
        )

    return f"<section><h2>Charts</h2>{''.join(blocks)}</section>"


def _changes(transformations: list[dict]) -> str:
    if not transformations:
        return ""

    items = "".join(
        f"<li>{html.escape(str(change.get('detail', '')))}</li>" for change in transformations
    )
    return (
        "<section><h2>What was changed before analysis</h2>"
        f'<ul class="changes">{items}</ul></section>'
    )


def _timing(log: list[dict]) -> str:
    """A proportional strip of where the run actually spent its time.

    Usually a surprise: charting dominates, and the statistics are nearly free.
    """
    if not log:
        return ""

    total = sum(entry.get("duration_ms", 0) for entry in log) or 1

    segments = "".join(
        f'<span style="width:{entry.get("duration_ms", 0) / total * 100:.2f}%;'
        f'background:{AGENT_COLOURS.get(entry.get("agent"), "#8FA3B4")}"></span>'
        for entry in log
    )

    rows = "".join(
        f'<div class="timing-row">'
        f'<span class="swatch" style="background:{AGENT_COLOURS.get(entry.get("agent"), "#8FA3B4")}"></span>'
        f"<span>{html.escape(str(entry.get('agent', '')))}</span>"
        f'<span class="ms">{entry.get("duration_ms", 0):,} ms</span>'
        f'<span class="pct">{entry.get("duration_ms", 0) / total * 100:.0f}%</span>'
        f"</div>"
        for entry in log
    )

    return (
        "<section><h2>Where the time went</h2>"
        f'<div class="timing"><div class="timing-bar">{segments}</div>'
        f'<div class="timing-key">{rows}</div>'
        f'<p class="note">{total:,} ms total across {len(log)} agents.</p></div></section>'
    )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _classify(finding: str) -> tuple[str, bool]:
    """Extract a short statistical label, and whether it needs attention.

    Returns (label, is_flagged). Flagged findings are data-quality warnings
    rather than results, and get the second signal colour.
    """
    correlation = re.search(r"r = (-?\d*\.?\d+)", finding)
    if correlation:
        return f"r = {correlation.group(1)}", False

    p_value = re.search(r"p = ([\d.e-]+)", finding)
    if p_value:
        return f"p = {p_value.group(1)}", False

    if "anomalous" in finding:
        return "isolation forest", True

    if "not normally distributed" in finding:
        return "shapiro-wilk", True

    if "explains" in finding and "variance" in finding:
        return "random forest", False

    return "observed", False


def _markdown_to_html(text: str) -> str:
    """Convert the small subset of markdown the Reporter actually emits.

    Headings, bold, italic, numbered and bulleted lists, paragraphs. A full
    markdown library would be a dependency for four rules.
    """
    lines = text.strip().split("\n")
    out: list[str] = []
    list_open = False

    def close_list() -> None:
        nonlocal list_open
        if list_open:
            out.append("</ol>")
            list_open = False

    for raw in lines:
        line = raw.strip()

        if not line:
            close_list()
            continue

        if line.startswith("## "):
            close_list()
            out.append(f"<h2>{_inline(line[3:])}</h2>")
        elif line.startswith("# "):
            close_list()
            out.append(f"<h2>{_inline(line[2:])}</h2>")
        elif re.match(r"^\d+\.\s", line):
            if not list_open:
                out.append("<ol>")
                list_open = True
            # Computed outside the f-string: backslashes there need Python 3.12
            item = re.sub(r"^\d+\.\s", "", line)
            out.append(f"<li>{_inline(item)}</li>")
        elif line.startswith(("- ", "* ")):
            if not list_open:
                out.append("<ol>")
                list_open = True
            out.append(f"<li>{_inline(line[2:])}</li>")
        else:
            close_list()
            out.append(f"<p>{_inline(line)}</p>")

    close_list()
    return "\n".join(out)


def _inline(text: str) -> str:
    """Escape, then re-enable bold, italic, and code spans."""
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<em>\1</em>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<code>\1</code>", escaped)
    return escaped
