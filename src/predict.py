from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features import prepare_model_input


def load_model_artifacts(
    models_dir: str | Path = "models",
):
    """Load the trained model, feature schema, and model configuration."""

    directory = Path(models_dir)

    model = joblib.load(
        directory / "final_xgboost_supply_stress.pkl"
    )
    feature_names = json.loads(
        (directory / "model_features.json").read_text(
            encoding="utf-8"
        )
    )
    config = json.loads(
        (directory / "model_config.json").read_text(
            encoding="utf-8"
        )
    )

    return model, feature_names, config


def predict_supply_stress(
    recent_data: pd.DataFrame,
    *,
    models_dir: str | Path = "models",
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Engineer features from recent history and produce risk probabilities.

    The input must contain enough prior history for each item-store series
    to calculate one-day, seven-day, and seven-day rolling features.
    """

    model, feature_names, config = load_model_artifacts(models_dir)

    if threshold is None:
        threshold = float(config["default_threshold"])

    scored_rows, X_ready, _ = prepare_model_input(
        recent_data,
        expected_features=feature_names,
    )

    if X_ready.empty:
        raise ValueError(
            "No rows contained enough history and complete source data "
            "to construct every required feature."
        )

    probabilities = model.predict_proba(X_ready)[:, 1]

    results = scored_rows.copy()
    results["stress_probability"] = probabilities
    results["stress_prediction"] = (
        probabilities >= threshold
    ).astype("int8")
    results["risk_label"] = results["stress_prediction"].map(
        {0: "No Stress", 1: "Stress Risk"}
    )

    return results
