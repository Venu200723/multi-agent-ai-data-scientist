"""End-to-end and unit tests for the multi-agent pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from multi_agent_ds.orchestrator import Orchestrator
from multi_agent_ds.agents.cleaning import CleaningAgent
from multi_agent_ds.agents.eda import EDAAgent
from multi_agent_ds.agents.ml import MLAgent
from multi_agent_ds.agents.explain import ExplainAgent
from multi_agent_ds.agents.report import ReportAgent
from multi_agent_ds import cli as cli_mod

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample.csv"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    assert SAMPLE.exists(), f"Missing sample data at {SAMPLE}"
    return pd.read_csv(SAMPLE)


def test_sample_csv_exists_and_has_churn(sample_df: pd.DataFrame):
    assert len(sample_df) >= 100
    assert "churn" in sample_df.columns


def test_cleaning_agent_reduces_nulls(sample_df: pd.DataFrame):
    agent = CleaningAgent()
    result = agent.run(sample_df, target="churn")
    cleaned = result["df"]
    assert cleaned.isna().sum().sum() == 0
    assert cleaned.shape[0] <= sample_df.shape[0]
    assert "actions" in result


def test_eda_agent_produces_profile_and_figures(sample_df: pd.DataFrame, tmp_path: Path):
    cleaned = CleaningAgent().run(sample_df, target="churn")["df"]
    result = EDAAgent(output_dir=tmp_path).run(cleaned, target="churn")
    assert result["profile"]["n_rows"] == cleaned.shape[0]
    assert result["profile"]["n_cols"] == cleaned.shape[1]
    assert len(result["figures"]) >= 1
    assert all(Path(f).exists() for f in result["figures"])


def test_ml_agent_classification_metrics(sample_df: pd.DataFrame):
    cleaned = CleaningAgent().run(sample_df, target="churn")["df"]
    result = MLAgent(random_state=0).run(cleaned, target="churn")
    assert result["task"] == "classification"
    assert "accuracy" in result["metrics"]
    assert 0.0 <= result["metrics"]["accuracy"] <= 1.0


def test_ml_agent_regression(tmp_path: Path):
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
            "price": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        }
    )
    result = MLAgent(random_state=0, test_size=0.25).run(df, target="price")
    assert result["task"] == "regression"
    assert "r2" in result["metrics"]
    assert "mae" in result["metrics"]


def test_explain_agent_offline_narrative(sample_df: pd.DataFrame, tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cleaned = CleaningAgent().run(sample_df, target="churn")["df"]
    ml_res = MLAgent(random_state=0).run(cleaned, target="churn")
    explain = ExplainAgent(output_dir=tmp_path, random_state=0).run(ml_res)
    assert len(explain["importances"]) >= 1
    assert "Offline summary" in explain["narrative"] or "metrics" in explain["narrative"].lower()
    assert any(Path(f).exists() for f in explain["figures"])


def test_report_agent_writes_md_and_html(tmp_path: Path):
    results = {
        "cleaning": {
            "original_shape": (10, 3),
            "cleaned_shape": (10, 3),
            "actions": ["noop"],
        },
        "eda": {
            "profile": {
                "n_rows": 10,
                "n_cols": 3,
                "n_numeric": 2,
                "n_categorical": 1,
                "missing_pct": 0.0,
            },
            "top_correlations": [],
            "figures": [],
            "actions": [],
        },
        "ml": {
            "target": "y",
            "task": "classification",
            "n_train": 7,
            "n_test": 3,
            "metrics": {"accuracy": 0.9},
            "actions": [],
        },
        "explain": {
            "narrative": "test narrative",
            "importances": [{"feature": "a", "importance": 0.1, "std": 0.01}],
            "figures": [],
            "actions": [],
        },
    }
    out = tmp_path / "r.md"
    res = ReportAgent(output_path=out).run(results, csv_name="t.csv")
    assert out.exists()
    assert out.read_text(encoding="utf-8").startswith("# Multi-Agent")
    assert Path(res["html_path"]).exists()
    assert "<html" in Path(res["html_path"]).read_text(encoding="utf-8").lower()


def test_orchestrator_guess_target():
    df = pd.DataFrame({"a": [1, 2], "churn": [0, 1]})
    assert Orchestrator._guess_target(df) == "churn"
    # "y" is in the preferred-name list
    df2 = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
    assert Orchestrator._guess_target(df2) == "y"
    # fallback: last column when nothing preferred matches
    df3 = pd.DataFrame({"alpha": [1], "beta": [2], "gamma": [3]})
    assert Orchestrator._guess_target(df3) == "gamma"


def test_full_pipeline(tmp_path: Path):
    report = tmp_path / "report.md"
    artifacts = tmp_path / "artifacts"
    orch = Orchestrator(artifacts_dir=artifacts, report_path=report)
    result = orch.run(SAMPLE, target="churn", report_path=report)

    assert result["target"] == "churn"
    assert result["ml"]["task"] == "classification"
    assert "accuracy" in result["ml"]["metrics"]
    assert report.exists()
    assert report.read_text(encoding="utf-8").startswith("# Multi-Agent")
    assert Path(result["report"]["html_path"]).exists()
    assert len(result["explain"]["importances"]) >= 1
    assert (artifacts / "eda_histograms.png").exists()
    assert (artifacts / "feature_importance.png").exists()


def test_cli_runs_on_sample(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # copy sample into temp so relative paths work cleanly
    csv = tmp_path / "sample.csv"
    csv.write_text(SAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
    report = tmp_path / "out.md"
    code = cli_mod.main([str(csv), "--target", "churn", "--report", str(report), "--artifacts", str(tmp_path / "art")])
    assert code == 0
    assert report.exists()
    captured = capsys.readouterr().out
    assert "run complete" in captured.lower() or "Metrics" in captured


def test_streamlit_app_imports():
    from multi_agent_ds import streamlit_app

    assert callable(streamlit_app.main)
