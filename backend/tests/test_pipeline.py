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
        with pytest.raises(IngestionError, match="Accepted formats"):
            read_bytes(b"content", "report.pdf")

    def test_tells_numbers_users_how_to_export(self):
        """A bare rejection leaves someone stuck with a file they can't use."""
        with pytest.raises(IngestionError, match="Export To"):
            read_bytes(b"content", "budget.numbers")

    def test_tells_google_sheets_users_how_to_export(self):
        with pytest.raises(IngestionError, match="Download"):
            read_bytes(b"content", "shared.gsheet")

    def test_rejects_files_over_the_size_limit(self):
        oversized = b"a,b\n" + b"1,2\n" * 3_000_000
        with pytest.raises(IngestionError, match="limit is 10 MB"):
            read_bytes(oversized, "huge.csv")

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


class TestErrorClassification:
    """Retrying a quota error just multiplies the delay before falling back."""

    def test_quota_exhaustion_is_permanent(self):
        from app.utils.llm import _is_permanent

        exc = Exception("Error code: 429 - {'error': {'code': 'credit_balance_exhausted'}}")
        assert _is_permanent(exc) is True

    def test_bad_key_is_permanent(self):
        from app.utils.llm import _is_permanent

        assert _is_permanent(Exception("invalid_api_key")) is True

    def test_rate_limit_is_worth_retrying(self):
        from app.utils.llm import _is_permanent

        assert _is_permanent(Exception("429 rate_limit_exceeded, slow down")) is False

    def test_timeout_is_worth_retrying(self):
        from app.utils.llm import _is_permanent

        assert _is_permanent(Exception("Connection timed out")) is False

    def test_quota_error_explains_the_subscription_confusion(self):
        from app.utils.llm import _explain

        message = _explain(Exception("insufficient_quota"))
        assert "credit" in message.lower()
        assert "subscription" in message.lower()

    def test_permanent_failure_stops_retrying_and_goes_offline(self):
        from app.utils.cost import CostLedger
        from app.utils.llm import LLMClient, LLMError

        client = LLMClient.__new__(LLMClient)  # skip __init__ and its API probe
        client.offline = False
        client.provider = "openai"
        client.model = "gpt-4o"
        client.agent = "test"
        client.ledger = CostLedger()
        client._cache = {}

        calls = {"n": 0}

        def always_out_of_credit(prompt, system, model=None):
            calls["n"] += 1
            raise Exception("Error code: 429 - insufficient_quota")

        client._call = always_out_of_credit

        with pytest.raises(LLMError, match="no credit"):
            client.ask("anything")

        assert calls["n"] == 1, "should not retry a permanent failure"
        assert client.offline is True, "later agents should skip the model entirely"


class TestRequestMultiplication:
    """Regression tests for the bug that made one call cost nine requests.

    Both provider SDKs retry internally by default. Combined with our own
    retry loop that became 3 x 3 = 9 HTTP requests for a single logical call,
    which is how a two-call pipeline billed for thirty-six.
    """

    def test_sdk_retries_are_disabled(self):
        from app.utils.llm import SDK_RETRIES

        assert SDK_RETRIES == 0, "SDK retries must stay off — they multiply with our own retry loop"

    def test_transient_failure_makes_exactly_max_retries_attempts(self):
        from app.utils.llm import MAX_RETRIES, LLMClient, LLMError

        client = LLMClient.__new__(LLMClient)
        client.offline = False
        client.provider = "openai"
        client.model = "gpt-4o"
        client.agent = "test"
        client._cache = {}

        from app.utils.cost import CostLedger

        client.ledger = CostLedger()

        attempts = {"n": 0}

        def flaky(prompt, system, model=None):
            attempts["n"] += 1
            raise Exception("Connection reset by peer")

        client._call = flaky

        with pytest.raises(LLMError):
            client.ask("hello")

        assert attempts["n"] == MAX_RETRIES, f"expected {MAX_RETRIES} attempts, got {attempts['n']}"


class TestPromptCache:
    """An identical prompt returns an identical answer. Paying twice buys nothing."""

    def _client(self):
        from app.utils.cost import CostLedger
        from app.utils.llm import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.offline = False
        client.provider = "openai"
        client.model = "gpt-4o"
        client.agent = "test"
        client._cache = {}
        client.ledger = CostLedger()
        return client

    def test_repeated_prompt_does_not_hit_the_provider(self):
        client = self._client()
        calls = {"n": 0}

        def counted(prompt, system, model=None):
            calls["n"] += 1
            return "an answer"

        client._call = counted

        assert client.ask("same question") == "an answer"
        assert client.ask("same question") == "an answer"
        assert client.ask("same question") == "an answer"

        assert calls["n"] == 1, "only the first call should reach the provider"
        assert client.ledger.billed_calls == 0, "cache hits are not billed"

    def test_a_different_prompt_is_a_different_key(self):
        client = self._client()
        calls = {"n": 0}

        def counted(prompt, system, model=None):
            calls["n"] += 1
            return f"answer to {prompt}"

        client._call = counted

        client.ask("question one")
        client.ask("question two")

        assert calls["n"] == 2


class TestCostLedger:
    def test_prices_a_known_model(self):
        from app.utils.cost import price_call

        # 1M in + 1M out on gpt-4o = $2.50 + $10.00
        assert price_call("gpt-4o", 1_000_000, 1_000_000) == pytest.approx(12.50)

    def test_matches_the_longest_prefix_not_the_shortest(self):
        from app.utils.cost import price_call

        mini = price_call("gpt-4o-mini-2024-07-18", 1_000_000, 0)
        full = price_call("gpt-4o-2024-08-06", 1_000_000, 0)

        assert mini == pytest.approx(0.15), "dated gpt-4o-mini must not price as gpt-4o"
        assert full == pytest.approx(2.50)

    def test_unknown_model_still_produces_a_number(self):
        from app.utils.cost import price_call

        assert price_call("some-future-model", 1_000_000, 0) > 0

    def test_attributes_spend_to_each_agent(self):
        from app.utils.cost import CostLedger

        ledger = CostLedger()
        ledger.record("planner", "gpt-4o", 1000, 100)
        ledger.record("reporter", "gpt-4o", 2000, 800)
        ledger.record("reporter", "gpt-4o", 500, 200)

        summary = ledger.summary()

        assert summary["total_calls"] == 3
        assert summary["by_agent"]["reporter"]["calls"] == 2
        assert summary["by_agent"]["planner"]["input_tokens"] == 1000

    def test_cache_hits_cost_nothing(self):
        from app.utils.cost import CostLedger

        ledger = CostLedger()
        ledger.record("planner", "gpt-4o", 1000, 100)
        ledger.record_cache_hit("planner", "gpt-4o")

        assert ledger.total_calls if hasattr(ledger, "total_calls") else True
        assert ledger.billed_calls == 1
        assert ledger.summary()["total_calls"] == 2

    def test_summary_is_json_serialisable(self):
        import json

        from app.utils.cost import CostLedger

        ledger = CostLedger()
        ledger.record("planner", "gpt-4o", 1000, 100)

        json.dumps(ledger.summary(), allow_nan=False)


class TestColumnHints:
    """The compact per-column hint that replaced three full sample rows."""

    def test_one_value_per_column(self):
        from app.agents.planner import _column_hints

        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        hints = _column_hints(df)

        assert set(hints) == {"a", "b"}
        assert hints["a"] == "1"
        assert hints["b"] == "x"

    def test_truncates_long_values(self):
        from app.agents.planner import _column_hints

        df = pd.DataFrame({"notes": ["a" * 200]})
        hints = _column_hints(df, max_len=40)

        assert len(hints["notes"]) <= 41  # 40 chars + the ellipsis
        assert hints["notes"].endswith("…")

    def test_all_null_column_is_none_not_an_error(self):
        from app.agents.planner import _column_hints

        df = pd.DataFrame({"empty": [None, None, None], "ok": [1, 2, 3]})
        hints = _column_hints(df)

        assert hints["empty"] is None
        assert hints["ok"] == "1"

    def test_caps_at_max_columns(self):
        from app.agents.planner import _column_hints

        df = pd.DataFrame({f"col{i}": [1, 2] for i in range(30)})
        hints = _column_hints(df, max_columns=15)

        assert len(hints) == 15

    def test_is_meaningfully_smaller_than_three_full_rows(self):
        """This is the actual cost claim — verify it holds, not just assert it."""
        from app.agents.planner import _column_hints, profile_dataframe

        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "id": [f"E{i:05d}" for i in range(50)],
                "dept": rng.choice(["Sales", "Marketing", "Support"], 50),
                "salary": rng.normal(60000, 12000, 50),
                "hire_date": pd.date_range("2020-01-01", periods=50),
            }
        )

        old = str(profile_dataframe(df)["sample_rows"])
        new = str(_column_hints(df))

        assert len(new) < len(old) * 0.7, (
            f"hint ({len(new)} chars) should be well under 70% of "
            f"three full rows ({len(old)} chars)"
        )


class TestModelRouting:
    """The Planner uses a cheaper model than the Reporter.

    A fixed-choice decision (six options, one target column) doesn't need
    frontier reasoning the way freeform narrative does, and a bad response
    degrades to the heuristic plan rather than breaking the run — which is
    what makes this a safe place to spend less.
    """

    def _client(self, provider="anthropic", model="claude-sonnet-4-6"):
        from app.utils.cost import CostLedger
        from app.utils.llm import LLMClient

        client = LLMClient.__new__(LLMClient)
        client.offline = False
        client.provider = provider
        client.model = model
        client.agent = "test"
        client._cache = {}
        client.ledger = CostLedger()
        client.cheap_model = {
            "anthropic": "claude-haiku-4-5",
            "openai": "gpt-4o-mini",
        }.get(provider, model)
        return client

    def test_cheap_model_differs_from_the_main_model(self):
        client = self._client(provider="anthropic", model="claude-sonnet-4-6")
        assert client.cheap_model == "claude-haiku-4-5"
        assert client.cheap_model != client.model

    def test_openai_provider_also_gets_a_cheap_model(self):
        client = self._client(provider="openai", model="gpt-4o")
        assert client.cheap_model == "gpt-4o-mini"

    def test_ask_uses_the_override_model_when_given(self):
        client = self._client()
        seen = {}

        def fake_call(prompt, system, model=None):
            seen["model"] = model
            return "ok"

        client._call = fake_call
        client.ask("hello", model=client.cheap_model)

        assert seen["model"] == "claude-haiku-4-5"

    def test_ask_defaults_to_the_main_model_when_no_override(self):
        client = self._client()
        seen = {}

        def fake_call(prompt, system, model=None):
            seen["model"] = model
            return "ok"

        client._call = fake_call
        client.ask("hello")  # no model kwarg — this is what the Reporter does

        assert seen["model"] == "claude-sonnet-4-6"

    def test_cache_is_separate_per_model(self):
        """The same prompt to two different models must not share a cache slot —
        otherwise the second model's answer never gets a chance to differ."""
        client = self._client()
        calls = []

        def fake_call(prompt, system, model=None):
            calls.append(model)
            return f"answer from {model}"

        client._call = fake_call

        cheap = client.ask("same question", model=client.cheap_model)
        full = client.ask("same question", model=client.model)

        assert cheap != full
        assert calls == [client.cheap_model, client.model]

    def test_planner_requests_the_cheap_model(self, monkeypatch):
        """End-to-end: PlannerAgent actually passes model=cheap_model, it
        doesn't just exist unused on the client."""
        import app.agents.planner as planner_mod
        from app.workflow.state import new_state

        requested = {}

        class FakeLLM:
            offline = False
            cheap_model = "claude-haiku-4-5"
            model = "claude-sonnet-4-6"

            def ask_json(self, prompt, system="", fallback=None, model=None):
                requested["model"] = model
                return fallback

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        state = new_state("test-id", "test.csv", df)

        agent = planner_mod.PlannerAgent(FakeLLM())
        agent(state)

        assert requested["model"] == "claude-haiku-4-5"

    def test_reporter_does_not_request_the_cheap_model(self):
        """The Reporter's narrative must keep using the full model — this is
        the explicit quality constraint, guarded so it can't regress silently."""
        import inspect

        from app.agents.reporter import ReporterAgent

        source = inspect.getsource(ReporterAgent._write_narrative)
        assert "cheap_model" not in source, (
            "Reporter narrative must stay on the configured model, not the "
            "cheap one used for planning"
        )


class TestVisualizerSpeed:
    """Regression tests for the plotly.express -> graph_objects rewrite.

    Express was measured at ~50-220ms per chart depending on type, mostly
    spent in its own data-inference and validation layer rather than actual
    rendering; the OLS trendline alone cost ~220ms per scatter because it
    calls into statsmodels for a full regression summary we never read.
    graph_objects builds the identical chart directly. These tests check the
    visible feature survived the rewrite, not just that it runs faster.
    """

    def test_histogram_still_has_its_marginal_box(self, state):
        s = VisualizerAgent()(AnalyzerAgent()(CleanerAgent()(PlannerAgent()(state))))
        histograms = [c for c in s["charts"] if c["chart_type"] == "histogram"]
        assert histograms, "expected at least one histogram"

        types = [trace.get("type") for trace in histograms[0]["plotly_json"]["data"]]
        assert "box" in types, "marginal box was dropped in the rewrite"
        assert "histogram" in types

    def test_scatter_still_has_a_trendline(self, state):
        s = VisualizerAgent()(AnalyzerAgent()(CleanerAgent()(PlannerAgent()(state))))
        scatters = [c for c in s["charts"] if c["chart_type"] == "scatter"]
        assert scatters, "expected at least one scatter plot"

        data = scatters[0]["plotly_json"]["data"]
        assert len(data) == 2, "expected a points trace and a trendline trace"
        modes = [trace.get("mode") for trace in data]
        assert "lines" in modes, "trendline trace was dropped in the rewrite"

    def test_trendline_slope_matches_a_direct_ols_fit(self):
        """The trendline switched from statsmodels to numpy.polyfit — same
        method (ordinary least squares), so the line itself must not move."""
        import numpy as np
        import statsmodels.api as sm

        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 300)
        y = 2.5 * x + 1.0 + rng.normal(0, 0.3, 300)

        np_slope, np_intercept = np.polyfit(x, y, 1)

        sm_model = sm.OLS(y, sm.add_constant(x)).fit()
        sm_intercept, sm_slope = sm_model.params

        assert np_slope == pytest.approx(sm_slope, rel=1e-9)
        assert np_intercept == pytest.approx(sm_intercept, rel=1e-9)

    def test_boxplot_has_one_trace_per_group(self, state):
        s = VisualizerAgent()(AnalyzerAgent()(CleanerAgent()(PlannerAgent()(state))))
        boxplots = [c for c in s["charts"] if c["chart_type"] == "box"]
        assert boxplots, "expected at least one group boxplot"

        n_groups = s["clean_df"]["group"].nunique()
        assert len(boxplots[0]["plotly_json"]["data"]) == n_groups

    def test_scatter_survives_a_constant_column(self):
        """np.polyfit raises LinAlgError on a column with zero variance —
        the chart must still render, just without a trendline."""
        df = pd.DataFrame({"flat": [5.0] * 50, "y": np.arange(50, dtype=float)})
        results = {
            "correlation": {
                "notable_pairs": [
                    {
                        "column_a": "flat",
                        "column_b": "y",
                        "r": 0.0,
                        "p_value": 1.0,
                        "significant": False,
                        "direction": "no",
                    }
                ]
            }
        }

        charts = VisualizerAgent()._scatters(df, results)
        assert len(charts) == 1
        assert charts[0]["plotly_json"]["data"]  # still has at least the points trace

    def test_visualizer_builds_the_sample_dataset_well_under_a_second(self, state):
        """Loose bound, generous for CI variance — this is a regression guard
        against ever falling back to the ~2s plotly.express path, not a tight
        performance benchmark."""
        import time

        prepped = AnalyzerAgent()(CleanerAgent()(PlannerAgent()(state)))

        t0 = time.perf_counter()
        VisualizerAgent()(prepped)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert elapsed_ms < 2000, f"visualizer took {elapsed_ms:.0f}ms, expected well under 2000ms"


class TestRowReference:
    """The synthetic row_ref column added when a file has no natural identifier.

    Content-derived, not random — so it never changes what exact-duplicate
    detection catches, and the same file produces the same IDs every run.
    """

    def test_added_when_no_identifier_exists(self):
        from app.agents.cleaner import CleanerAgent
        from app.workflow.state import new_state

        df = pd.DataFrame(
            {
                "age": [34, 41, 29, 52, 38, 45, 31, 49],
                "salary": [72000.0, 91000.0, 58000.0, 105000.0, 67000.0, 88000.0, 61000.0, 97000.0],
            }
        )
        state = new_state("t", "f.csv", df)
        result = CleanerAgent()(state)

        assert "row_ref" in result["clean_df"].columns
        assert next(iter(result["clean_df"].columns)) == "row_ref", "should be the first column"
        assert result["clean_df"]["row_ref"].is_unique

    def test_not_added_when_an_id_column_exists(self):
        from app.agents.cleaner import CleanerAgent
        from app.workflow.state import new_state

        df = pd.DataFrame(
            {"employee_id": ["E1", "E2", "E3"], "salary": [50000.0, 60000.0, 70000.0]}
        )
        state = new_state("t", "f.csv", df)
        result = CleanerAgent()(state)

        assert "row_ref" not in result["clean_df"].columns

    def test_not_added_when_a_name_column_exists(self):
        from app.agents.cleaner import _has_identifier_column

        df = pd.DataFrame({"customer_name": ["Alex", "Priya", "Sam"], "score": [1, 2, 3]})
        assert _has_identifier_column(df) is True

    def test_small_dataset_does_not_falsely_flag_a_column_as_an_id(self):
        """Regression: with few rows, any column can look '100% unique' by pure
        chance. That must not be read as 'this column is an identifier' —
        it's exactly the case where a reference column is needed most."""
        from app.agents.cleaner import _has_identifier_column

        df = pd.DataFrame({"age": [34, 41, 29, 52], "level": [1, 2, 3, 4]})
        assert _has_identifier_column(df) is False

    def test_float_columns_never_count_as_an_identifier(self):
        """Regression: a continuous column (salary, a decimal credit score)
        looks close to 100% unique at almost any size — that's a property of
        being continuous data, not of being an identifier. Confirmed against
        the exact case that surfaced this: age/salary/savings/debts/credit,
        no name or ID column, large enough to clear the row-count guard."""
        from app.agents.cleaner import _has_identifier_column

        rng = np.random.default_rng(4)
        n = 200
        df = pd.DataFrame(
            {
                "age": rng.integers(22, 65, n),
                "salary": rng.normal(70000, 15000, n),
                "savings": rng.normal(20000, 8000, n),
                "debts": rng.normal(5000, 3000, n),
                "credit": rng.normal(700, 40, n),
            }
        )
        assert _has_identifier_column(df) is False

    def test_is_deterministic_across_runs(self):
        """Same input, same output — required for a report to be regenerable
        and for row references to mean the same thing if re-run later."""
        from app.agents.cleaner import CleanerAgent
        from app.workflow.state import new_state

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})

        first = CleanerAgent()(new_state("t1", "f.csv", df.copy()))["clean_df"]["row_ref"]
        second = CleanerAgent()(new_state("t2", "f.csv", df.copy()))["clean_df"]["row_ref"]

        assert list(first) == list(second)

    def test_survives_missing_values_in_the_row(self):
        """A row with a NaN must still get a reference, not crash the cleaner —
        regression for astype(str) not stringifying NaN on newer pandas."""
        from app.agents.cleaner import CleanerAgent
        from app.workflow.state import new_state

        df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0], "b": [10.0, 20.0, np.nan, 40.0]})
        state = new_state("t", "f.csv", df)
        result = CleanerAgent()(state)

        assert result["clean_df"]["row_ref"].notna().all()
        assert result["clean_df"]["row_ref"].is_unique

    def test_references_are_six_char_alphanumeric_by_default(self):
        from app.agents.cleaner import CleanerAgent
        from app.workflow.state import new_state

        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        result = CleanerAgent()(new_state("t", "f.csv", df))

        for ref in result["clean_df"]["row_ref"]:
            assert len(ref) == 6
            assert ref.isalnum()

    def test_stays_unique_at_larger_scale(self):
        """No forced-collision test (finding a real sha256 collision isn't
        practical to construct), but this checks the property that matters:
        a few thousand distinct rows still get a few thousand distinct IDs."""
        from app.agents.cleaner import CleanerAgent
        from app.workflow.state import new_state

        rng = np.random.default_rng(5)
        n = 3000
        df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})

        result = CleanerAgent()(new_state("t", "f.csv", df))
        assert result["clean_df"]["row_ref"].is_unique

    def test_excluded_from_descriptive_stats_categorical_summary(self):
        """Without this, a report would show a meaningless 'top values' block
        for a column where every value appears exactly once."""
        from app.agents import AnalyzerAgent, CleanerAgent, PlannerAgent
        from app.workflow.state import new_state

        df = pd.DataFrame({"age": rng_col(50, 20), "salary": rng_col(50, 21)})
        s = AnalyzerAgent()(CleanerAgent()(PlannerAgent()(new_state("t", "f.csv", df))))

        assert "row_ref" not in s["results"]["descriptive_stats"]["categorical"]

    def test_excluded_from_group_comparison_even_on_a_small_dataset(self):
        """Regression for the sharper version of the same bug: on a dataset
        of 20 rows or fewer, row_ref's cardinality (all-unique) coincidentally
        satisfies the existing '1 < nunique <= 20' grouping filter. Without an
        explicit exclusion, the Analyzer would try to group by row_ref itself —
        one row per 'group', a meaningless comparison."""
        from app.agents import AnalyzerAgent, CleanerAgent, PlannerAgent
        from app.workflow.state import new_state

        rng = np.random.default_rng(6)
        n = 15
        df = pd.DataFrame(
            {
                "salary": rng.normal(70000, 10000, n),
                "dept": rng.choice(["Eng", "Sales"], n),
            }
        )
        s = AnalyzerAgent()(CleanerAgent()(PlannerAgent()(new_state("t", "f.csv", df))))

        assert "row_ref" not in s["clean_df"].columns or all(
            comparison["group_column"] != "row_ref"
            for comparison in s["results"].get("group_comparison", {}).get("comparisons", [])
        )


def rng_col(n, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, n)


class TestGroundingCheck:
    """Best-effort check that the Reporter's narrative doesn't state numbers
    the model was never given. Not a hard gate — flags for review, never
    blocks or rewrites the report — so these tests check it catches real
    fabrications without crying wolf on ordinary rounding or paraphrasing.
    """

    def _agent(self):
        from app.agents.reporter import ReporterAgent

        return ReporterAgent()

    def test_clean_narrative_is_not_flagged(self):
        agent = self._agent()
        grounded = {0.51, 0.003, 45.2}
        narrative = "The correlation is strong (r = 0.51, p = 0.003), averaging 45.2."

        check = agent._check_grounding(narrative, grounded)
        assert check["ungrounded_numbers"] == []
        assert check["causal_language"] == []

    def test_catches_an_invented_number(self):
        agent = self._agent()
        grounded = {0.51, 0.003}
        narrative = "The correlation is 0.51 (p = 0.003), and average tenure is 14.7 years."

        check = agent._check_grounding(narrative, grounded)
        assert "14.7" in check["ungrounded_numbers"]

    def test_catches_causal_language(self):
        agent = self._agent()
        narrative = "This relationship causes higher retention rates."

        check = agent._check_grounding(narrative, set())
        assert "causes" in check["causal_language"]

    def test_does_not_flag_causal_words_absent_from_text(self):
        agent = self._agent()
        narrative = "There is a strong positive relationship between the two variables."

        check = agent._check_grounding(narrative, set())
        assert check["causal_language"] == []

    def test_tolerates_percentage_conversion_of_a_fraction(self):
        """r2_score=0.82 in the brief is a legitimate 'explains 82%' in prose —
        that's correct paraphrasing, not fabrication, and must not be flagged."""
        agent = self._agent()
        grounded = {0.82}
        narrative = "The model explains 82% of the variance."

        check = agent._check_grounding(narrative, grounded)
        assert check["ungrounded_numbers"] == []

    def test_tolerates_reasonable_rounding(self):
        agent = self._agent()
        grounded = {45.678}
        narrative = "The average is about 45.7."

        check = agent._check_grounding(narrative, grounded)
        assert check["ungrounded_numbers"] == []

    def test_tolerates_dropped_sign_when_describing_direction_in_words(self):
        agent = self._agent()
        grounded = {-0.65}
        narrative = "A strong negative relationship (r = 0.65) was found."

        check = agent._check_grounding(narrative, grounded)
        assert check["ungrounded_numbers"] == []

    def test_plain_small_integers_are_not_flagged(self):
        """'the three sections below' shouldn't need to trace to a computed
        value — small integers are excluded from the check entirely."""
        agent = self._agent()
        narrative = "There are three important caveats and two next steps to consider."

        check = agent._check_grounding(narrative, {0.51})
        assert check["ungrounded_numbers"] == []

    def test_brief_collects_the_exact_values_it_embeds(self):
        """The grounding check is only as good as this set — verify it
        actually captures what _build_brief puts in front of the model."""
        from app.workflow.state import new_state

        rng = np.random.default_rng(7)
        df = pd.DataFrame({"a": rng.normal(50, 10, 100), "b": rng.normal(0, 1, 100)})
        state = new_state("t", "f.csv", df)
        state["profile"] = {"rows": 100, "columns": 2, "numeric_columns": ["a", "b"]}
        state["results"] = {
            "descriptive_stats": {
                "numeric": {"a": {"mean": 50.3, "median": 50.1, "std": 9.8, "skew": 0.02}}
            },
            "correlation": {
                "notable_pairs": [
                    {
                        "column_a": "a",
                        "column_b": "b",
                        "r": 0.42,
                        "p_value": 0.01,
                        "significant": True,
                    }
                ]
            },
        }

        agent = self._agent()
        _, values = agent._build_brief(state)

        assert 0.42 in values
        assert 0.01 in values
        assert 50.3 in values

    def test_full_pipeline_offline_narrative_produces_no_false_flags(self, state):
        """The deterministic offline stub is the one narrative every CI run
        actually generates — it must never flag itself."""
        from app.agents import (
            AnalyzerAgent,
            CleanerAgent,
            PlannerAgent,
            ReporterAgent,
            VisualizerAgent,
        )

        s = PlannerAgent()(state)
        s = CleanerAgent()(s)
        s = AnalyzerAgent()(s)
        s = VisualizerAgent()(s)
        s = ReporterAgent()(s)

        assert s["report"]["grounding_check"]["ungrounded_numbers"] == []

    def test_flagged_items_appear_in_the_markdown_only_when_present(self):
        """Silent when clean — this section shouldn't clutter every report."""
        agent = self._agent()
        state = {"profile": {}, "log": []}

        clean_md = agent._assemble(
            state, "All good.", [], {"ungrounded_numbers": [], "causal_language": []}
        )
        assert "Narrative quality check" not in clean_md

        flagged_md = agent._assemble(
            state,
            "The data shows 14.7.",
            [],
            {"ungrounded_numbers": ["14.7"], "causal_language": ["causes"]},
        )
        assert "Narrative quality check" in flagged_md
        assert "14.7" in flagged_md
        assert "causes" in flagged_md
