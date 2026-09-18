# Multi-Agent Autonomous AI Data Scientist

**Flagship MVP** — a modular Python package that runs an end-to-end tabular data science pipeline with specialized "agents" for cleaning, EDA, ML, explainability, and reporting.

## Problem

Analysts spend hours wiring the same CSV → clean → explore → model → explain → report loop. This project turns that loop into a **composable multi-agent pipeline** you can run from the CLI (or Streamlit) on any tabular CSV.

## Architecture

```
CSV
 └─► CleaningAgent   (nulls, dtypes, outliers, duplicates)
      └─► EDAAgent   (profile, correlations, plots)
           └─► MLAgent  (sklearn RF + holdout metrics)
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
| `multi_agent_ds/streamlit_app.py` | Optional UI |

Offline by default: if `OPENAI_API_KEY` is unset (or the call fails), ExplainAgent writes a deterministic summary.

## How to run

```bash
cd multi-agent-ai-data-scientist
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# tests
python -m pytest

# CLI on bundled telecom-churn-style sample
python -m multi_agent_ds.cli data/sample.csv
# or with explicit target / report path
python -m multi_agent_ds.cli data/sample.csv --target churn --report examples/sample_report.md

# Streamlit UI
streamlit run multi_agent_ds/streamlit_app.py
```

**Outputs**

- `examples/sample_report.md` (+ `.html`) — pipeline report
- `artifacts/eda_histograms.png`, `eda_correlation.png`, `feature_importance.png`

## Sample data

`data/sample.csv` — synthetic but realistic customer churn table (~400 rows): tenure, charges, contract, internet, support flags, binary `churn`.

## Resume bullets

- Built a **multi-agent tabular DS pipeline** (clean → EDA → sklearn ML → permutation XAI → Markdown/HTML report) runnable via CLI and Streamlit.
- Designed modular agent interfaces with an orchestrator, offline LLM fallback, and pytest coverage on synthetic churn data.
- Delivered holdout metrics + feature-importance narratives without requiring cloud credentials.

## Suggested GitHub repo name

`multi-agent-ai-data-scientist`
