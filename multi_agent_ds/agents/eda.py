"""EDAAgent — profile, correlations, simple plots."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class EDAAgent:
    """Produce a compact EDA profile and optional figures."""

    output_dir: str | Path | None = None
    log: list[str] = field(default_factory=list)

    def run(self, df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
        self.log = []
        out = Path(self.output_dir) if self.output_dir else None
        if out:
            out.mkdir(parents=True, exist_ok=True)

        numeric = df.select_dtypes(include=[np.number])
        categorical = df.select_dtypes(exclude=[np.number])

        profile = {
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "n_numeric": int(numeric.shape[1]),
            "n_categorical": int(categorical.shape[1]),
            "missing_pct": float(df.isna().mean().mean() * 100),
            "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        }

        describe = numeric.describe().round(4).to_dict() if not numeric.empty else {}
        corr = {}
        top_corr_pairs: list[tuple[str, str, float]] = []
        if numeric.shape[1] >= 2:
            corr_df = numeric.corr(numeric_only=True)
            corr = corr_df.round(4).to_dict()
            pairs = []
            cols = corr_df.columns.tolist()
            for i, a in enumerate(cols):
                for b in cols[i + 1 :]:
                    v = corr_df.loc[a, b]
                    if pd.notna(v):
                        pairs.append((a, b, float(v)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_corr_pairs = pairs[:5]
            self.log.append(f"Computed correlations; top pair={top_corr_pairs[0] if top_corr_pairs else 'n/a'}")

        target_summary: dict[str, Any] = {}
        if target and target in df.columns:
            if pd.api.types.is_numeric_dtype(df[target]):
                target_summary = {
                    "type": "numeric",
                    "mean": float(df[target].mean()),
                    "std": float(df[target].std()),
                    "min": float(df[target].min()),
                    "max": float(df[target].max()),
                }
            else:
                vc = df[target].value_counts(normalize=True).round(4)
                target_summary = {"type": "categorical", "class_balance": vc.to_dict()}
            self.log.append(f"Profiled target '{target}'")

        figures: list[str] = []
        if out and not numeric.empty:
            # Histograms for up to 4 numeric cols
            cols = numeric.columns[:4].tolist()
            fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3))
            if len(cols) == 1:
                axes = [axes]
            for ax, col in zip(axes, cols):
                ax.hist(numeric[col].dropna(), bins=20, color="#4C78A8", edgecolor="white")
                ax.set_title(col, fontsize=9)
            fig.tight_layout()
            hist_path = out / "eda_histograms.png"
            fig.savefig(hist_path, dpi=120)
            plt.close(fig)
            figures.append(str(hist_path))
            self.log.append(f"Saved histograms → {hist_path.name}")

            if numeric.shape[1] >= 2:
                fig, ax = plt.subplots(figsize=(5, 4))
                cax = ax.imshow(numeric.corr(), cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric.columns)))
                ax.set_yticks(range(len(numeric.columns)))
                ax.set_xticklabels(numeric.columns, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(numeric.columns, fontsize=7)
                fig.colorbar(cax, ax=ax, fraction=0.046)
                fig.tight_layout()
                corr_path = out / "eda_correlation.png"
                fig.savefig(corr_path, dpi=120)
                plt.close(fig)
                figures.append(str(corr_path))
                self.log.append(f"Saved correlation heatmap → {corr_path.name}")

        cat_summary = {}
        for col in categorical.columns[:5]:
            cat_summary[col] = df[col].value_counts().head(5).to_dict()

        return {
            "agent": "eda",
            "profile": profile,
            "describe": describe,
            "correlation": corr,
            "top_correlations": top_corr_pairs,
            "target_summary": target_summary,
            "categorical_top": cat_summary,
            "figures": figures,
            "actions": list(self.log),
        }
