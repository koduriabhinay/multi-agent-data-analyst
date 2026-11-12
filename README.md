# Multi-Agent Data Analyst

Upload a CSV or Excel file and a pipeline of five agents profiles it, cleans it, runs statistics,
builds charts and writes up the findings.

The numbers come from scipy and scikit-learn (hypothesis tests, correlations, Isolation Forest,
random forest feature importance). The language model chooses which analyses to run and writes the
narrative around the computed results; it does not produce any of the figures.

```
CSV/Excel -> Planner -> Cleaner -> Analyzer -> Visualizer -> Reporter -> Markdown report + charts
```

The pipeline is a LangGraph workflow, served by FastAPI and also runnable from the command line.

## Quickstart

Requires Python 3.12.

```bash
git clone https://github.com/koduriabhinay/multi-agent-data-analyst.git
cd multi-agent-data-analyst

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r backend/requirements-dev.txt

cd backend
python scripts/make_sample_data.py              # writes CSVs to ../data_samples/
python -m app.cli ../data_samples/employees.csv
```

Without an API key the pipeline runs in offline mode: every statistic and chart is computed as
usual and only the written narrative is replaced by a stub. To enable the narrative, copy
`.env.example` to `.env` in the repository root and set `ANTHROPIC_API_KEY`.

## API

```bash
cd backend
uvicorn app.main:app --reload       # docs at http://localhost:8000/docs
```

```bash
# Upload a file; returns an id immediately and runs the pipeline in the background
curl -F "file=@../data_samples/sales.csv" http://localhost:8000/api/analyses

# Fetch the finished report
curl http://localhost:8000/api/analyses/<id>/report
```

| Endpoint | Description |
|---|---|
| `POST /api/analyses` | Upload a file and start the pipeline |
| `GET /api/analyses` | List recent analyses |
| `GET /api/analyses/{id}` | Full results: profile, statistics, charts, report |
| `GET /api/analyses/{id}/report` | Report as Markdown |
| `DELETE /api/analyses/{id}` | Delete an analysis |
| `WS /api/ws/{id}` | Progress updates while the pipeline runs |

## Agents

**Planner** profiles the dataframe with pandas (shape, dtypes, null rates, low-cardinality
categorical columns) and asks the model which analyses to run. If the model is unavailable or
returns an invalid plan, a heuristic plan built from the same profile is used.

**Cleaner** drops columns that are more than 60% empty, removes exact duplicate rows, converts text
columns that hold dates or numbers, and imputes missing values (median for numeric, mode for
categorical). Outliers are flagged but kept. Every transformation is recorded and listed in the
report.

**Analyzer** runs the planned analyses: descriptive statistics with skew and kurtosis, Pearson
correlation with p-values, Shapiro-Wilk normality tests, t-tests and ANOVA across groups, Isolation
Forest anomaly detection, and a random forest with feature importance when a suitable target column
exists. Each analysis runs independently, so one failure does not discard the others.

**Visualizer** builds Plotly figures based on what the Analyzer found, for example a correlation
heatmap only when correlations were computed and scatter plots only for significant pairs. Figures
are stored as JSON so a frontend can render them interactively.

**Reporter** assembles the facts in Python, sends the model a compact brief and receives prose
back. Key findings are extracted directly from the computed results, so they are present and
correct in offline mode too.

## Design notes

- **The model plans and writes; Python computes.** Every number in the report comes from scipy or
  scikit-learn, which keeps token usage low and makes the output checkable.
- **Offline mode is tested.** CI runs the full pipeline without secrets and asserts on the
  statistical findings.
- **Failures are contained per agent.** `BaseAgent.__call__` catches exceptions, records them in the
  run log and lets the pipeline return partial results.
- **SQLite by default.** Results are stored in JSON columns, so the same schema works on PostgreSQL
  by changing `DATABASE_URL`.

## Tests

```bash
cd backend
pytest tests/ -v
pytest tests/ --cov=app
ruff check . && ruff format --check .
```

Tests run without an API key. Several use a dataset with a planted correlation and known defects
and assert that the pipeline finds exactly those and nothing spurious.

## Docker

```bash
docker compose up --build
```

This starts the simple web server in `app/web_main.py` on port 8000, which exposes a single
synchronous `POST /api/analyze` upload endpoint and serves a static UI if `backend/static/` exists.
Export `ANTHROPIC_API_KEY` first to enable the narrative.

## VS Code

The `.vscode/` folder contains launch configurations (API server, CLI on the sample data, CLI on
the open file, all tests, current test file) and tasks for installing dependencies, generating
sample data, testing, linting and starting the server. Formatting and import sorting use Ruff.

## Configuration

All settings have defaults except the API key. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | Unset means offline mode |
| `LLM_PROVIDER` | `anthropic` | Or `openai` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Model used for the Reporter narrative |
| `PLANNER_MODEL` | provider default | Smaller model for the Planner (Haiku or gpt-4o-mini) |
| `DATABASE_URL` | `sqlite:///./analyst.db` | PostgreSQL URLs also work |
| `LOG_LEVEL` | `INFO` | |

## Project layout

```
backend/
  app/
    agents/       planner, cleaner, analyzer, visualizer, reporter, shared base class
    workflow/     pipeline state and LangGraph graph
    data/         file ingestion and validation
    db/           SQLAlchemy models
    api/          FastAPI routes and service layer
    export/       HTML report export
    utils/        LLM client with retries and offline fallback, cost tracking
    cli.py        run an analysis without the server
    main.py       FastAPI application
    web_main.py   minimal upload server used by the Docker image
  scripts/        sample data generator
  tests/
.vscode/          debug configurations and tasks
.github/          CI workflow (lint, format check, tests)
Dockerfile, docker-compose.yml
```

## License

MIT, see [LICENSE](LICENSE).
