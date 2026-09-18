"""Orchestrator — run cleaning → EDA → ML → explain → report in sequence."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .agents.cleaning import CleaningAgent
from .agents.eda import EDAAgent
from .agents.explain import ExplainAgent
from .agents.ml import MLAgent
from .agents.report import ReportAgent


@dataclass
class Orchestrator:
    """End-to-end multi-agent tabular DS pipeline."""

    artifacts_dir: str | Path = "artifacts"
    report_path: str | Path = "examples/sample_report.md"
    test_size: float = 0.25
    random_state: int = 42
    history: list[str] = field(default_factory=list)

    def run(
        self,
        csv_path: str | Path,
        target: str | None = None,
        report_path: str | Path | None = None,
    ) -> dict[str, Any]:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        artifacts = Path(self.artifacts_dir)
        artifacts.mkdir(parents=True, exist_ok=True)
        report_out = Path(report_path) if report_path else Path(self.report_path)

        df = pd.read_csv(csv_path)
        self.history.append(f"Loaded {csv_path} shape={df.shape}")

        if target is None:
            target = self._guess_target(df)
            self.history.append(f"Auto-selected target='{target}'")

        cleaning = CleaningAgent()
        clean_res = cleaning.run(df, target=target)
        df_clean: pd.DataFrame = clean_res["df"]
        self.history.extend(clean_res["actions"])

        eda = EDAAgent(output_dir=artifacts)
        eda_res = eda.run(df_clean, target=target)
        self.history.extend(eda_res["actions"])

        ml = MLAgent(test_size=self.test_size, random_state=self.random_state)
        ml_res = ml.run(df_clean, target=target)
        self.history.extend(ml_res["actions"])

        explain = ExplainAgent(output_dir=artifacts, random_state=self.random_state)
        explain_res = explain.run(ml_res)
        self.history.extend(explain_res["actions"])

        # Strip non-serializable bits for report assembly
        ml_for_report = {k: v for k, v in ml_res.items() if k not in {"pipeline", "X_test", "y_test"}}
        results = {
            "cleaning": {k: v for k, v in clean_res.items() if k != "df"},
            "eda": eda_res,
            "ml": ml_for_report,
            "explain": explain_res,
        }

        report = ReportAgent(output_path=report_out)
        report_res = report.run(results, csv_name=csv_path.name)
        self.history.extend(report_res["actions"])

        return {
            "target": target,
            "cleaning": clean_res,
            "eda": eda_res,
            "ml": ml_res,
            "explain": explain_res,
            "report": report_res,
            "history": list(self.history),
        }

    @staticmethod
    def _guess_target(df: pd.DataFrame) -> str:
        preferred = [
            "churn",
            "target",
            "label",
            "y",
            "default",
            "converted",
            "outcome",
            "salary",
            "price",
            "revenue",
        ]
        lower_map = {c.lower(): c for c in df.columns}
        for name in preferred:
            if name in lower_map:
                return lower_map[name]
        # last column heuristic
        return df.columns[-1]
