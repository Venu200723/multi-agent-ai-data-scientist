# Multi-Agent Autonomous AI Data Scientist

**Flagship MVP** — a modular Python package that runs an end-to-end tabular data science pipeline with specialized agents for cleaning, EDA, ML, explainability, and reporting.

> Demo-ready: an interviewer can clone, install, and run the Streamlit app or CLI in **under 5 minutes**. No paid APIs required.

## Problem

Analysts spend hours wiring the same CSV → clean → explore → model → explain → report loop. This project turns that loop into a **composable multi-agent pipeline** you can run from the CLI or a polished Streamlit UI on any tabular CSV.

## Architecture

```
CSV
 └─► CleaningAgent   (nulls, dtypes, outliers, duplicates)
      └─► EDAAgent   (profile, correlations, plots)
           └─► MLAgent  (sklearn RandomForest + holdout metrics)
                └─► ExplainAgent  (permutation importance + optional LLM narrative)
                     └─► ReportAgent  (Markdown + HTML)
```

| Module | Role |
|---|---|
| `multi_agent_ds/agents/cleaning.py` | Hygiene & imputation |
| `multi_agent_ds/agents/eda.py` | Profiling & matplotlib plots |
| `multi_agent_ds/agents/ml.py` | Train/test RandomForest pipeline |
| `multi_agent_ds/agents/explain.py` | Permutation importance; `OPENAI_API_KEY` optional |
| `multi_agent_ds/agents/report.py` | Markdown / HTML assembly |
| `multi_agent_ds/orchestrator.py` | Sequential runner |
| `multi_agent_ds/cli.py` | CLI entrypoint |
| `multi_agent_ds/streamlit_app.py` | Polished Streamlit UI |

**Offline by default:** if `OPENAI_API_KEY` is unset (or the call fails), ExplainAgent writes a deterministic summary. Core demo never requires a paid API.

## Quickstart (< 5 minutes)

```bash
cd multi-agent-ai-data-scientist
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# tests (should be green)
python -m pytest -q

# CLI on bundled telecom-churn sample
python -m multi_agent_ds.cli data/sample.csv
# or with explicit target / report path
python -m multi_agent_ds.cli data/sample.csv --target churn --report examples/sample_report.md

# Streamlit UI (recommended for interviews)
streamlit run multi_agent_ds/streamlit_app.py
```

### Docker (optional)

```bash
docker build -t multi-agent-ds .
docker run --rm -p 8501:8501 multi-agent-ds
# open http://localhost:8501
```

## Outputs

| Path | Description |
|---|---|
| `examples/sample_report.md` | Markdown pipeline report |
| `examples/sample_report.html` | HTML twin of the report |
| `artifacts/eda_histograms.png` | EDA histograms |
| `artifacts/eda_correlation.png` | Correlation heatmap |
| `artifacts/feature_importance.png` | Permutation importance chart |

## Sample data

`data/sample.csv` — synthetic but realistic customer churn table (~400 rows): tenure, charges, contract, internet, support flags, binary `churn`.

## Screenshots / UI placeholders

> Replace these with real screenshots before publishing the GitHub repo.

| Placeholder | What to capture |
|---|---|
| `docs/screenshots/01-upload.png` | Streamlit upload + sample checkbox + target picker |
| `docs/screenshots/02-metrics.png` | Holdout metrics cards after a successful run |
| `docs/screenshots/03-explain.png` | Narrative + importance table + charts |
| `docs/screenshots/04-report.png` | Download buttons + markdown preview |

```
docs/screenshots/
  01-upload.png      ← (placeholder) data preview & target select
  02-metrics.png     ← (placeholder) accuracy / f1 / roc_auc cards
  03-explain.png     ← (placeholder) feature importance + narrative
  04-report.png      ← (placeholder) report download panel
```

## Interview demo script (~3 minutes)

**Setup (before the call):** `streamlit run multi_agent_ds/streamlit_app.py` already open on localhost.

1. **Hook (15s):** “This is a multi-agent tabular DS pipeline — five specialized agents orchestrated end-to-end, offline-first.”
2. **Show architecture (20s):** Point at the sidebar agent list / README diagram.
3. **Run (30s):** Leave “Use bundled sample.csv” checked → target `churn` → click **Run pipeline**. Narrate: cleaning nulls/dupes → EDA plots → RandomForest holdout → permutation XAI → report.
4. **Results (60s):** Walk through metrics cards, explanation narrative, importance chart, and download MD/HTML.
5. **CLI (20s):** Switch to terminal: `python -m multi_agent_ds.cli data/sample.csv --target churn` — same pipeline, CI-friendly.
6. **Close (20s):** “No paid APIs for the core path; OpenAI is optional behind an env var with offline fallback. Pytest covers the full pipeline.”

**What to say if asked about agents:** Each agent is a small dataclass with a `run()` method; the orchestrator sequences them and collects an action log — intentional simplicity for interview clarity, not a heavyweight framework.

## Resume bullets

- Built a **multi-agent tabular DS pipeline** (clean → EDA → sklearn ML → permutation XAI → Markdown/HTML report) runnable via CLI and Streamlit.
- Designed modular agent interfaces with an orchestrator, offline LLM fallback, and pytest coverage on synthetic churn data.
- Delivered holdout metrics + feature-importance narratives without requiring cloud credentials.
- Packaged a demo-ready MVP with sample data, pre-generated artifacts, Dockerfile, and a sub-5-minute interviewer quickstart.

## Project layout

```
multi-agent-ai-data-scientist/
├── multi_agent_ds/
│   ├── agents/          # cleaning, eda, ml, explain, report
│   ├── orchestrator.py
│   ├── cli.py
│   └── streamlit_app.py
├── data/sample.csv
├── examples/            # pre-generated reports
├── artifacts/           # pre-generated plots
├── tests/
├── Dockerfile
├── requirements.txt
└── pyproject.toml
```

## Suggested GitHub repo name

`multi-agent-ai-data-scientist`

## License

MIT (or your preferred license when publishing).
