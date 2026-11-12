"""
Tests for the HTML report exporter.

The important checks here are escaping and structure: a filename containing a
quote or a script tag must not be able to break out of the document, and every
chart must reach the page as parseable JSON.
"""

from __future__ import annotations

import json
from html.parser import HTMLParser

import numpy as np
import pandas as pd
import pytest

from app.export.html_report import _classify, _markdown_to_html, render_report
from app.workflow.pipeline import run_pipeline_sync
from app.workflow.state import new_state


class TagBalance(HTMLParser):
    """Minimal well-formedness check — catches unclosed and crossed tags."""

    VOID = {"meta", "link", "br", "hr", "img", "input", "source", "area"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.mismatched: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.stack and self.stack[-1] == tag:
            self.stack.pop()
        elif tag in self.stack:
            self.mismatched.append(tag)
            self.stack.remove(tag)


@pytest.fixture
def analysed():
    rng = np.random.default_rng(3)
    n = 150
    x = rng.normal(20, 4, n)
    df = pd.DataFrame(
        {
            "x": x,
            "y": x * 3 + rng.normal(0, 2, n),
            "group": rng.choice(["north", "south"], n),
        }
    )
    return run_pipeline_sync(new_state("t", "sample.csv", df))


class TestStructure:
    def test_produces_a_complete_document(self, analysed):
        page = render_report(analysed)
        assert page.startswith("<!doctype html>")
        assert page.rstrip().endswith("</html>")

    def test_tags_are_balanced(self, analysed):
        parser = TagBalance()
        parser.feed(render_report(analysed))

        assert parser.stack == [], f"unclosed: {parser.stack}"
        assert parser.mismatched == [], f"mismatched: {parser.mismatched}"

    def test_includes_every_section(self, analysed):
        page = render_report(analysed)

        for heading in ("Key findings", "Charts", "Where the time went"):
            assert f"<h2>{heading}</h2>" in page

    def test_embeds_one_container_per_chart(self, analysed):
        page = render_report(analysed)
        # +2 for the two references in the bootstrap script
        assert page.count("data-figure") == len(analysed["charts"]) + 2

    def test_reports_the_real_shape(self, analysed):
        page = render_report(analysed)
        assert f"{analysed['profile']['rows']:,}" in page


class TestEscaping:
    """A filename is untrusted input — it comes from whatever the user uploaded."""

    def test_script_tag_in_filename_is_neutralised(self):
        df = pd.DataFrame({"a": range(40), "b": range(40)})
        state = run_pipeline_sync(new_state("t", "<script>alert('xss')</script>.csv", df))

        page = render_report(state)
        assert "<script>alert" not in page
        assert "&lt;script&gt;" in page

    def test_quote_in_filename_cannot_break_an_attribute(self):
        df = pd.DataFrame({"a": range(40), "b": range(40)})
        state = run_pipeline_sync(new_state("t", 'evil" onload="x.csv', df))

        page = render_report(state)
        assert 'onload="x' not in page

    def test_chart_json_survives_attribute_encoding(self, analysed):
        """The figure must still parse after being escaped into an attribute."""
        import re

        page = render_report(analysed)
        match = re.search(r'data-figure="([^"]*)"', page)
        assert match is not None

        import html as html_module

        figure = json.loads(html_module.unescape(match.group(1)))
        assert "data" in figure


class TestFindingLabels:
    def test_labels_a_correlation_with_its_coefficient(self):
        label, flagged = _classify("x and y move together (positive, r = 0.94, p = 0).")
        assert label == "r = 0.94"
        assert flagged is False

    def test_labels_a_group_test_with_its_p_value(self):
        label, _ = _classify("y differs across group (One-way ANOVA, p = 0.003).")
        assert label == "p = 0.003"

    def test_flags_anomalies_for_attention(self):
        label, flagged = _classify("15 rows (5.0%) look anomalous compared with the rest.")
        assert flagged is True
        assert "isolation" in label

    def test_flags_non_normal_distributions(self):
        _, flagged = _classify("4 column(s) are not normally distributed (x, y).")
        assert flagged is True

    def test_falls_back_rather_than_crashing(self):
        label, _ = _classify("Something entirely unexpected happened.")
        assert label == "observed"


class TestMarkdownConversion:
    def test_converts_headings(self):
        assert "<h2>Summary</h2>" in _markdown_to_html("## Summary")

    def test_converts_bold_and_italic(self):
        out = _markdown_to_html("This is **bold** and *slanted*.")
        assert "<strong>bold</strong>" in out
        assert "<em>slanted</em>" in out

    def test_converts_numbered_lists(self):
        out = _markdown_to_html("1. First\n2. Second")
        assert out.count("<li>") == 2
        assert "<ol>" in out and "</ol>" in out

    def test_closes_lists_before_the_next_heading(self):
        out = _markdown_to_html("1. Item\n\n## Next section")
        assert out.index("</ol>") < out.index("<h2>")

    def test_escapes_html_inside_markdown(self):
        out = _markdown_to_html("A <script>bad()</script> line")
        assert "<script>" not in out

    def test_handles_empty_input(self):
        assert _markdown_to_html("") == ""


class TestTimingStrip:
    def test_widths_sum_to_one_hundred_percent(self, analysed):
        import re

        page = render_report(analysed)
        widths = [float(w) for w in re.findall(r"width:([\d.]+)%", page)]

        assert widths, "no timing segments rendered"
        assert sum(widths) == pytest.approx(100, abs=0.5)

    def test_names_every_agent_that_ran(self, analysed):
        page = render_report(analysed)
        for entry in analysed["log"]:
            assert entry["agent"] in page


class TestDegradedInput:
    """The exporter runs on whatever the pipeline produced, including failures."""

    def test_survives_an_empty_state(self):
        page = render_report({})
        assert page.startswith("<!doctype html>")

    def test_survives_a_state_with_no_charts(self, analysed):
        analysed["charts"] = []
        page = render_report(analysed)
        assert "<h2>Charts</h2>" not in page
        assert page.rstrip().endswith("</html>")

    def test_survives_a_missing_narrative(self, analysed):
        analysed["report"]["narrative"] = ""
        page = render_report(analysed)
        assert "<h2>Key findings</h2>" in page
