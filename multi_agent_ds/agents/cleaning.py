"""CleaningAgent — missing values, dtypes, outliers, basic hygiene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CleaningAgent:
    """Clean a tabular DataFrame for downstream EDA / ML."""

    drop_threshold: float = 0.6
    iqr_factor: float = 3.0
    log: list[str] = field(default_factory=list)

    def run(self, df: pd.DataFrame, target: str | None = None) -> dict[str, Any]:
        self.log = []
        original_shape = df.shape
        work = df.copy()

        # Drop near-empty columns
        null_frac = work.isna().mean()
        drop_cols = null_frac[null_frac > self.drop_threshold].index.tolist()
        if drop_cols:
            work = work.drop(columns=drop_cols)
            self.log.append(f"Dropped high-null columns: {drop_cols}")

        # Infer / coerce numerics where mostly numeric
        for col in work.columns:
            if col == target:
                continue
            if work[col].dtype == object:
                coerced = pd.to_numeric(work[col], errors="coerce")
                if coerced.notna().mean() > 0.8:
                    work[col] = coerced
                    self.log.append(f"Coerced '{col}' to numeric")

        # Impute
        for col in work.columns:
            if work[col].isna().any():
                if pd.api.types.is_numeric_dtype(work[col]):
                    fill = work[col].median()
                    work[col] = work[col].fillna(fill)
                    self.log.append(f"Imputed numeric '{col}' with median={fill:.4g}")
                else:
                    mode = work[col].mode(dropna=True)
                    fill = mode.iloc[0] if len(mode) else "unknown"
                    work[col] = work[col].fillna(fill)
                    self.log.append(f"Imputed categorical '{col}' with mode='{fill}'")

        # Soft outlier capping (IQR) on numeric features only
        numeric_cols = work.select_dtypes(include=[np.number]).columns.tolist()
        if target and target in numeric_cols:
            numeric_cols = [c for c in numeric_cols if c != target]
        capped = 0
        for col in numeric_cols:
            q1, q3 = work[col].quantile(0.25), work[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr
            before = ((work[col] < lo) | (work[col] > hi)).sum()
            if before:
                work[col] = work[col].clip(lo, hi)
                capped += int(before)
        if capped:
            self.log.append(f"Capped {capped} extreme outlier values (IQR×{self.iqr_factor})")

        # Drop duplicate rows
        n_dup = int(work.duplicated().sum())
        if n_dup:
            work = work.drop_duplicates().reset_index(drop=True)
            self.log.append(f"Removed {n_dup} duplicate rows")

        if not self.log:
            self.log.append("No cleaning actions required")

        return {
            "agent": "cleaning",
            "df": work,
            "original_shape": original_shape,
            "cleaned_shape": work.shape,
            "actions": list(self.log),
            "dropped_columns": drop_cols,
        }
