# Multi-Agent Data Analyst

Upload a spreadsheet. Five agents profile it, clean it, run statistics, build charts, and write up what they found.

The statistics are real — scipy hypothesis tests, sklearn models, Isolation Forest anomaly detection. The language model decides *which* analyses are worth running and writes the narrative. It never invents a number.

```
CSV/Excel  →  Planner  →  Cleaner  →  Analyzer  →  Visualizer  →  Reporter  →  Markdown + charts
```

---

## Run it in 60 seconds

```bash
git clone <your-repo-url> && cd multi-agent-data-analyst

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt

cd backend
python scripts/make_sample_data.py
python -m app.cli ../data_samples/employees.csv
```

That prints a full analysis report. **No API key needed** — without one the pipeline runs in offline mode, where every statistic and chart is still real and only the written narrative is stubbed.

To turn the narrative on:

```bash
cp .env.example .env      # then add your ANTHROPIC_API_KEY
```

---

## Run the API

```bash
cd backend
uvicorn app.main:app --reload
```

Interactive docs at <http://localhost:8000/docs>.

```bash
# Upload a file — returns immediately with an ID
curl -F "file=@../data_samples/sales.csv" http://localhost:8000/api/analyses

# Fetch the finished report
curl http://localhost:8000/api/analyses/<id>/report
```

| Endpoint | What it does |
|---|---|
| `POST /api/analyses` | Upload a file, start the pipeline |
| `GET /api/analyses` | List recent analyses |
| `GET /api/analyses/{id}` | Full results: profile, stats, charts, report |
| `GET /api/analyses/{id}/report` | Report as raw markdown |
| `DELETE /api/analyses/{id}` | Remove an analysis |
| `WS /api/ws/{id}` | Live progress while the pipeline runs |

---

## What each agent does

**Planner** profiles the dataframe in pandas — shape, dtypes, null rates, which categorical columns have few enough values to group by — then asks the model which analyses to run. If the model is unavailable or returns something unusable, a heuristic plan derived from the same profile takes over.

**Cleaner** drops columns more than 60% empty, removes exact duplicates, converts text columns that are really dates or numbers, and fills gaps (median for numbers, mode for categories). Outliers are flagged but the rows stay — extreme values are often the interesting part. Every change is recorded and appears in the final report.

**Analyzer** runs whatever the plan asked for: descriptive statistics with skew and kurtosis, Pearson correlation with p-values, Shapiro-Wilk normality tests, t-tests and ANOVA across groups, Isolation Forest for anomalies, and a random forest with feature importance when there's a sensible target column. Each analysis is isolated, so one failure doesn't lose the others.

**Visualizer** builds Plotly figures driven by what the Analyzer actually found — a heatmap only if correlations were computed, scatter plots only for pairs that cleared significance. Charts are stored as JSON, so the frontend renders them interactively.

**Reporter** assembles the facts in Python, hands the model a compact brief, and gets prose back. Key findings are extracted deterministically from the numbers, so they're correct even when the model is offline.

---

## Working in VS Code

Open the repo root. Install the recommended extensions when prompted.

**Run and Debug** (`F5`) has five configurations ready:

- *API server (reload)* — debug the server with breakpoints
- *CLI: analyse employees.csv* — step through the whole pipeline
- *CLI: analyse current file* — run against whatever CSV you have open
- *Tests: all* / *Tests: current file*

**Tasks** (`Cmd/Ctrl+Shift+P` → "Run Task"): install dependencies, generate sample data, run tests, lint and format, start the server.

Format-on-save and import sorting are wired to Ruff. Tests appear in the Testing sidebar automatically.

---

## Testing

```bash
cd backend
pytest tests/ -v                       # 35 tests
pytest tests/ --cov=app                 # with coverage
ruff check . && ruff format --check .   # lint
```

Tests run without an API key. Several use a dataset with a *planted* correlation and known defects, then assert the pipeline finds exactly those and doesn't report spurious ones.

---

## Docker

```bash
docker compose up --build
```

Brings up the API on port 8000 with Postgres behind it. Set `ANTHROPIC_API_KEY` in your shell or `.env` first if you want the narrative.

---

## Project layout

```
backend/
  app/
    agents/       planner, cleaner, analyzer, visualizer, reporter, base
    workflow/     state definition, LangGraph pipeline
    data/         file ingestion and validation
    db/           SQLAlchemy models
    api/          FastAPI routes, service layer
    utils/        LLM client with retry and offline fallback
    cli.py        run an analysis without the server
    main.py       app entry point
  tests/          35 tests
  scripts/        sample data generator
.vscode/          debug configs, tasks, settings
.github/          CI workflow
```

---

## Configuration

Everything has a working default except the API key. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(unset)* | Blank means offline mode |
| `LLM_PROVIDER` | `anthropic` | Or `openai` |
| `LLM_MODEL` | `claude-sonnet-4-6` | |
| `DATABASE_URL` | `sqlite:///./analyst.db` | Postgres works unchanged |
| `LOG_LEVEL` | `INFO` | |

---

## Design decisions worth knowing

**SQLite by default.** The project runs with zero setup. Results are stored as JSON columns, which are portable to Postgres without a schema change.

**The model decides, Python computes.** Every number in the report comes from scipy or sklearn. The model chooses what to run and writes prose about the results. This keeps token usage low and makes the output verifiable.

**Offline mode is a first-class path, not a degraded one.** CI runs the full pipeline with no secrets and asserts real statistical findings.

**Failures are contained per-agent.** `BaseAgent.__call__` catches exceptions, records them in the run log, and lets the pipeline return partial results rather than a stack trace.
