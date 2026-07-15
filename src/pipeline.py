from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.features import prepare_model_input
from src.load_data import load_m5_data
from src.preprocess import build_analytical_table
from src.train import (
    chronological_split,
    save_model_artifacts,
    train_final_xgboost,
)


@dataclass
class TrainingResult:
    analytical_data: pd.DataFrame
    feature_data: pd.DataFrame
    model_features: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    model: object


def run_training_pipeline(
    data_dir: str | Path,
    *,
    max_items: int | None = 100,
    train_fraction: float = 0.80,
    models_dir: str | Path | None = None,
) -> TrainingResult:
    """Load raw files, engineer features, train XGBoost, and optionally save it."""

    raw = load_m5_data(data_dir)
    analytical = build_analytical_table(
        raw.sales,
        raw.calendar,
        raw.prices,
        max_items=max_items,
    )

    feature_data, X, feature_names = prepare_model_input(analytical)
    y = feature_data["stress_event"]

    X_train, X_test, y_train, y_test = chronological_split(
        X,
        y,
        train_fraction=train_fraction,
    )

    model = train_final_xgboost(X_train, y_train)

    if models_dir is not None:
        save_model_artifacts(
            model,
            feature_names,
            models_dir,
            default_threshold=0.50,
        )

    return TrainingResult(
        analytical_data=analytical,
        feature_data=feature_data,
        model_features=feature_names,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        model=model,
    )
