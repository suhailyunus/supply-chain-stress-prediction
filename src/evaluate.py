from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    classification_report,
    precision_recall_curve,
)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float


def classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict:
    """Return a structured classification report."""

    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0,
    )


def print_classification_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Print the familiar scikit-learn classification report."""

    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0,
        )
    )


def find_best_f1_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> ThresholdResult:
    """Find the probability threshold that maximizes F1."""

    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    f1 = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-10)
    )
    best_index = int(np.argmax(f1))

    return ThresholdResult(
        threshold=float(thresholds[best_index]),
        precision=float(precision[best_index]),
        recall=float(recall[best_index]),
        f1=float(f1[best_index]),
    )


def evaluate_probabilities(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.50,
) -> dict:
    """Summarize model ranking and threshold-based classification."""

    predictions = (probabilities >= threshold).astype(int)

    return {
        "threshold": threshold,
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "classification_report": classification_metrics(
            y_true,
            predictions,
        ),
    }


def plot_confusion_matrix(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    title: str = "Confusion Matrix",
    output_path: str | Path | None = None,
) -> None:
    """Plot and optionally save a confusion matrix."""

    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_precision_recall(
    y_true: pd.Series,
    probabilities: np.ndarray,
    *,
    title: str = "Precision–Recall Curve",
    output_path: str | Path | None = None,
) -> None:
    """Plot and optionally save the Precision–Recall curve."""

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )
    average_precision = average_precision_score(
        y_true,
        probabilities,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        label=f"Average Precision = {average_precision:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")

    plt.show()
