"""ExplainAgent — permutation importance (+ optional LLM narrative)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


@dataclass
class ExplainAgent:
    """Compute feature importances and a short narrative summary."""

    output_dir: str | Path | None = None
    n_repeats: int = 5
    random_state: int = 42
    log: list[str] = field(default_factory=list)

    def run(self, ml_result: dict[str, Any]) -> dict[str, Any]:
        self.log = []
        pipe = ml_result["pipeline"]
        X_test: pd.DataFrame = ml_result["X_test"]
        y_test = ml_result["y_test"]
        task = ml_result["task"]
        feature_names = list(X_test.columns)

        scoring = "accuracy" if task == "classification" else "r2"
        try:
            result = permutation_importance(
                pipe,
                X_test,
                y_test,
                n_repeats=self.n_repeats,
                random_state=self.random_state,
                scoring=scoring,
                n_jobs=-1,
            )
            importances = sorted(
                zip(feature_names, result.importances_mean, result.importances_std),
                key=lambda x: x[1],
                reverse=True,
            )
            self.log.append(f"Computed permutation importance ({scoring})")
        except Exception as exc:  # fallback to tree impurity if available
            self.log.append(f"Permutation importance failed ({exc}); using model fallback")
            model = pipe.named_steps["model"]
            if hasattr(model, "feature_importances_"):
                # Approximate with raw model importances (post-transform dims may differ)
                vals = model.feature_importances_
                names = [f"f{i}" for i in range(len(vals))]
                importances = sorted(zip(names, vals, [0.0] * len(vals)), key=lambda x: x[1], reverse=True)
            else:
                importances = [(n, 0.0, 0.0) for n in feature_names]

        top = importances[:10]
        figures: list[str] = []
        out = Path(self.output_dir) if self.output_dir else None
        if out and top:
            out.mkdir(parents=True, exist_ok=True)
            labels = [t[0] for t in reversed(top)]
            means = [t[1] for t in reversed(top)]
            stds = [t[2] for t in reversed(top)]
            fig, ax = plt.subplots(figsize=(6, max(2.5, 0.35 * len(labels))))
            ax.barh(labels, means, xerr=stds, color="#54A24B", alpha=0.85)
            ax.set_xlabel("Permutation importance")
            ax.set_title("Feature importance (holdout)")
            fig.tight_layout()
            path = out / "feature_importance.png"
            fig.savefig(path, dpi=120)
            plt.close(fig)
            figures.append(str(path))
            self.log.append(f"Saved importance plot → {path.name}")

        narrative = self._narrative(top, ml_result.get("metrics", {}), task)
        return {
            "agent": "explain",
            "importances": [
                {"feature": f, "importance": float(m), "std": float(s)} for f, m, s in top
            ],
            "narrative": narrative,
            "figures": figures,
            "actions": list(self.log),
        }

    def _narrative(
        self, top: list[tuple], metrics: dict[str, float], task: str
    ) -> str:
        offline = self._offline_summary(top, metrics, task)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            self.log.append("No OPENAI_API_KEY — using offline summary")
            return offline
        try:
            # Optional lightweight call; fail soft to offline
            from urllib import request
            import json

            prompt = (
                "Summarize this ML explanation in 3 short bullets for a non-technical reader.\n"
                f"Task: {task}\nMetrics: {metrics}\nTop features: {top[:5]}\n"
            )
            body = json.dumps(
                {
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 200,
                }
            ).encode()
            req = request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            text = data["choices"][0]["message"]["content"].strip()
            self.log.append("Generated LLM narrative via OpenAI")
            return text
        except Exception as exc:
            self.log.append(f"LLM call failed ({exc}); offline fallback")
            return offline

    @staticmethod
    def _offline_summary(
        top: list[tuple], metrics: dict[str, float], task: str
    ) -> str:
        metric_bits = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
        feats = ", ".join(f"{f} ({m:.3f})" for f, m, _ in top[:3]) or "n/a"
        return (
            f"Offline summary — {task} model holdout metrics: {metric_bits}. "
            f"Strongest drivers by permutation importance: {feats}. "
            "Use these features first when debugging predictions or collecting better data."
        )
