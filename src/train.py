from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split already chronologically ordered data into past and future."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_index = int(len(X) * train_fraction)

    return (
        X.iloc[:split_index].copy(),
        X.iloc[split_index:].copy(),
        y.iloc[:split_index].copy(),
        y.iloc[split_index:].copy(),
    )


def calculate_scale_pos_weight(y: pd.Series) -> float:
    """Calculate the majority-to-minority class ratio."""

    negative_count = int((y == 0).sum())
    positive_count = int((y == 1).sum())

    if positive_count == 0:
        raise ValueError("The positive class is absent from the training data.")

    return negative_count / positive_count


def build_random_forest(**overrides: Any) -> RandomForestClassifier:
    """Create the tuned Random Forest benchmark."""

    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 15,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def build_xgboost(
    *,
    scale_pos_weight: float,
    **overrides: Any,
) -> XGBClassifier:
    """Create the selected class-balanced XGBoost model."""

    params: dict[str, Any] = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.03,
        "min_child_weight": 3,
        "gamma": 0,
        "subsample": 0.9,
        "colsample_bytree": 1.0,
        "scale_pos_weight": scale_pos_weight,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    params.update(overrides)
    return XGBClassifier(**params)


def train_final_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Fit the selected model using class imbalance from the training set."""

    model = build_xgboost(
        scale_pos_weight=calculate_scale_pos_weight(y_train)
    )
    model.fit(X_train, y_train)
    return model


def save_model_artifacts(
    model: Any,
    feature_names: list[str],
    output_dir: str | Path,
    *,
    default_threshold: float = 0.50,
    alternative_threshold: float | None = None,
) -> None:
    """Persist the model, feature schema, and deployment configuration."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        model,
        directory / "final_xgboost_supply_stress.pkl",
    )

    (directory / "model_features.json").write_text(
        json.dumps(feature_names, indent=2),
        encoding="utf-8",
    )

    config = {
        "model_type": type(model).__name__,
        "positive_class": 1,
        "positive_class_label": "Supply Stress",
        "default_threshold": default_threshold,
        "alternative_threshold": alternative_threshold,
    }
    (directory / "model_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )
