"""MLAgent — train a simple sklearn model with holdout metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class MLAgent:
    """Train a baseline model and return metrics + fitted pipeline."""

    test_size: float = 0.25
    random_state: int = 42
    log: list[str] = field(default_factory=list)

    def run(self, df: pd.DataFrame, target: str) -> dict[str, Any]:
        self.log = []
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")

        y = df[target]
        X = df.drop(columns=[target])

        # Drop ID-like columns
        drop_ids = [c for c in X.columns if c.lower() in {"id", "index", "row_id"}]
        if drop_ids:
            X = X.drop(columns=drop_ids)
            self.log.append(f"Dropped id-like columns: {drop_ids}")

        task = self._infer_task(y)
        self.log.append(f"Inferred task: {task}")

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]

        transformers = []
        if num_cols:
            transformers.append(("num", StandardScaler(), num_cols))
        if cat_cols:
            transformers.append(
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    cat_cols,
                )
            )
        pre = ColumnTransformer(transformers, remainder="drop")

        if task == "classification":
            model = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )
        else:
            model = RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        pipe = Pipeline([("pre", pre), ("model", model)])

        stratify = y if task == "classification" and y.nunique() > 1 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics: dict[str, float] = {}

        if task == "classification":
            metrics["accuracy"] = float(accuracy_score(y_test, preds))
            metrics["f1_weighted"] = float(f1_score(y_test, preds, average="weighted"))
            if hasattr(pipe.named_steps["model"], "predict_proba") and y.nunique() == 2:
                try:
                    proba = pipe.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_test, proba))
                except Exception:
                    pass
        else:
            metrics["r2"] = float(r2_score(y_test, preds))
            metrics["mae"] = float(mean_absolute_error(y_test, preds))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_test, preds)))

        self.log.append(f"Holdout metrics: {metrics}")

        # Feature names after preprocessing (best-effort)
        feature_names = num_cols + cat_cols

        return {
            "agent": "ml",
            "task": task,
            "target": target,
            "pipeline": pipe,
            "metrics": metrics,
            "feature_names": feature_names,
            "X_test": X_test,
            "y_test": y_test,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "actions": list(self.log),
        }

    @staticmethod
    def _infer_task(y: pd.Series) -> str:
        if pd.api.types.is_numeric_dtype(y):
            nunique = y.nunique(dropna=True)
            if nunique <= 10 and set(y.dropna().unique()).issubset({0, 1, True, False}) or (
                nunique <= 8 and not pd.api.types.is_float_dtype(y)
            ):
                # small discrete set → classification
                if nunique <= 8:
                    return "classification"
            return "regression"
        return "classification"
