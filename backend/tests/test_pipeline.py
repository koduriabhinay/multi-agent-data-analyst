"""
Tests for the analysis pipeline.

These run without an API key — the LLM client falls back to offline mode, so
CI never needs a secret to verify the statistics and cleaning logic are right.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.agents import AnalyzerAgent, CleanerAgent, PlannerAgent, VisualizerAgent
from app.data.ingestion import IngestionError, read_bytes
from app.workflow.pipeline import run_pipeline_sync
from app.workflow.state import new_state


@pytest.fixture
def messy_df() -> pd.DataFrame:
    """A dataframe with a known correlation and known defects."""
    rng = np.random.default_rng(0)
    n = 200

    x = rng.normal(50, 10, n)
    df = pd.DataFrame(
        {
            "x": x,
            # y is x plus noise, so correlation should be strong and detectable
            "y": x * 2 + rng.normal(0, 5, n),
            "noise": rng.normal(0, 1, n),
            "group": rng.choice(["A", "B", "C"], n),
            "mostly_empty": [None] * n,
        }
    )

    df.loc[:9, "noise"] = np.nan  # 10 missing values
    return pd.concat([df, df.head(5)], ignore_index=True)  # 5 duplicates


@pytest.fixture
def state(messy_df):
    return new_state("test", "messy.csv", messy_df)


# ----------------------------------------------------------------------
# Ingestion
# ----------------------------------------------------------------------
class TestIngestion:
    def test_reads_csv(self):
        df = read_bytes(b"a,b\n1,2\n3,4\n", "data.csv")
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2

    def test_rejects_unsupported_extension(self):
        with pytest.raises(IngestionError, match="supported format"):
            read_bytes(b"content", "notes.txt")

    def test_rejects_empty_file(self):
        with pytest.raises(IngestionError, match="empty"):
            read_bytes(b"", "data.csv")

    def test_rejects_single_column(self):
        with pytest.raises(IngestionError, match="at least two"):
            read_bytes(b"only\n1\n2\n", "data.csv")

    def test_handles_latin1_encoding(self):
        # A CSV exported from Excel on Windows, with a non-UTF-8 byte
        content = "name,city\nJos\xe9,M\xe1laga\n".encode("latin-1")
        df = read_bytes(content, "data.csv")
        assert len(df) == 1

    def test_renames_unnamed_index_column(self):
        df = read_bytes(b",a,b\n0,1,2\n", "data.csv")
        assert not any(c.startswith("Unnamed:") for c in df.columns)


# ----------------------------------------------------------------------
# Individual agents
# ----------------------------------------------------------------------
class TestPlanner:
    def test_profiles_the_dataframe(self, state):
        result = PlannerAgent()(state)
        profile = result["profile"]

        assert profile["rows"] == 205
        assert "x" in profile["numeric_columns"]
        assert "group" in profile["categorical_columns"]
        assert profile["duplicate_rows"] == 5

    def test_produces_a_valid_plan(self, state):
        result = PlannerAgent()(state)
        analyses = result["plan"]["analyses"]

        assert "descriptive_stats" in analyses
        assert "correlation" in analyses  # there are 3 numeric columns

    def test_identifies_groupable_columns(self, state):
        result = PlannerAgent()(state)
        assert "group" in result["profile"]["groupable_columns"]


class TestCleaner:
    def test_drops_mostly_empty_columns(self, state):
        result = CleanerAgent()(PlannerAgent()(state))
        assert "mostly_empty" not in result["clean_df"].columns

    def test_removes_duplicates(self, state):
        result = CleanerAgent()(PlannerAgent()(state))
        assert len(result["clean_df"]) == 200

    def test_fills_missing_values(self, state):
        result = CleanerAgent()(PlannerAgent()(state))
        assert result["clean_df"].isnull().sum().sum() == 0

    def test_records_every_change(self, state):
        result = CleanerAgent()(PlannerAgent()(state))
        steps = {t["step"] for t in result["transformations"]}

        assert "drop_sparse_columns" in steps
        assert "drop_duplicates" in steps
        assert "impute_missing" in steps

    def test_leaves_the_original_untouched(self, state):
        original_rows = len(state["raw_df"])
        CleanerAgent()(PlannerAgent()(state))
        assert len(state["raw_df"]) == original_rows


class TestAnalyzer:
    @pytest.fixture
    def analyzed(self, state):
        s = PlannerAgent()(state)
        s = CleanerAgent()(s)
        return AnalyzerAgent()(s)

    def test_computes_descriptive_stats(self, analyzed):
        stats = analyzed["results"]["descriptive_stats"]["numeric"]
        assert "x" in stats
        assert stats["x"]["mean"] == pytest.approx(50, abs=3)

    def test_finds_the_planted_correlation(self, analyzed):
        pairs = analyzed["results"]["correlation"]["notable_pairs"]
        xy = next(p for p in pairs if {p["column_a"], p["column_b"]} == {"x", "y"})
        assert xy["r"] > 0.9
        assert xy["significant"] is True

    def test_does_not_report_spurious_correlations(self, analyzed):
        pairs = analyzed["results"]["correlation"]["notable_pairs"]
        # "noise" is independent, so it shouldn't pair strongly with anything
        assert not any("noise" in (p["column_a"], p["column_b"]) for p in pairs)

    def test_results_are_json_serialisable(self, analyzed):
        import json

        json.dumps(analyzed["results"])  # raises if numpy types leaked through


class TestVisualizer:
    def test_builds_charts(self, state):
        s = VisualizerAgent()(AnalyzerAgent()(CleanerAgent()(PlannerAgent()(state))))

        assert len(s["charts"]) > 0
        for chart in s["charts"]:
            assert chart["title"]
            assert chart["plotly_json"]["data"]

    def test_includes_a_correlation_heatmap(self, state):
        s = VisualizerAgent()(AnalyzerAgent()(CleanerAgent()(PlannerAgent()(state))))
        assert any(c["chart_type"] == "heatmap" for c in s["charts"])


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------
class TestPipeline:
    def test_runs_end_to_end(self, state):
        result = run_pipeline_sync(state)

        assert result["status"] == "completed"
        assert result["report"]["markdown"]
        assert len(result["charts"]) > 0

    def test_logs_every_agent(self, state):
        result = run_pipeline_sync(state)
        agents = [entry["agent"] for entry in result["log"]]

        assert agents == ["planner", "cleaner", "analyzer", "visualizer", "reporter"]
        assert all(entry["status"] == "success" for entry in result["log"])

    def test_report_mentions_the_real_correlation(self, state):
        result = run_pipeline_sync(state)
        findings = " ".join(result["report"]["key_findings"])
        assert "x" in findings and "y" in findings

    def test_survives_a_single_column_of_constants(self):
        df = pd.DataFrame({"constant": [1] * 50, "varies": range(50)})
        result = run_pipeline_sync(new_state("t", "flat.csv", df))
        assert result["status"] == "completed"

    def test_survives_an_all_text_dataframe(self):
        df = pd.DataFrame({"a": ["x"] * 30, "b": ["y"] * 30})
        result = run_pipeline_sync(new_state("t", "text.csv", df))
        assert result["status"] == "completed"


class TestSerializationHelper:
    def test_converts_nan_to_none(self):
        from app.utils.serialization import json_safe

        assert json_safe(float("nan")) is None
        assert json_safe(float("inf")) is None

    def test_converts_numpy_scalars(self):
        from app.utils.serialization import json_safe

        assert json_safe(np.int64(5)) == 5
        assert json_safe(np.float64(2.5)) == 2.5
        assert json_safe(np.bool_(True)) is True

    def test_converts_timestamps_to_iso_strings(self):
        from app.utils.serialization import json_safe

        result = json_safe(pd.Timestamp("2024-03-01"))
        assert result.startswith("2024-03-01")

    def test_recurses_into_nested_structures(self):
        import json

        from app.utils.serialization import json_safe

        messy = {"a": [np.float64("nan"), pd.Timestamp("2024-01-01")], "b": {"c": np.int32(3)}}
        json.dumps(json_safe(messy), allow_nan=False)  # raises if anything leaked
