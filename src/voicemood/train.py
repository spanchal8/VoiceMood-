"""Training driver: takes a feature matrix, runs the standard protocol.

Protocol (justified in docs/methodology.md):
    1. Stratified train/test split (test_size = 0.2, seed fixed).
    2. Fit classifier on training portion.
    3. Predict on test set.
    4. Compute headline metrics (accuracy, macro F1) with bootstrap CIs.
    5. Save fitted model, predictions, and per-class metrics to results/.

The function returns a dict of artifacts so downstream evaluation (McNemar,
calibration) can be done across models without retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .config import (
    BOOTSTRAP_CI,
    BOOTSTRAP_RESAMPLES,
    EMOTION_LABELS,
    METRICS_DIR,
    RANDOM_SEED,
    TEST_SIZE,
)
from .models import ModelName, build_classifier


@dataclass
class TrainedArtifact:
    """Everything we need from one trained model for downstream analysis."""

    model_name: ModelName
    feature_name: str            # "mfcc", "wav2vec2", or "yamnet"
    pipeline: Pipeline           # fitted sklearn pipeline
    X_test: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.feature_name}__{self.model_name}"


def _bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = BOOTSTRAP_CI,
    seed: int = RANDOM_SEED,
) -> tuple[float, float, float]:
    """Compute point estimate and percentile bootstrap CI for accuracy.

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = float(accuracy_score(y_true, y_pred))

    boots = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        boots[i] = accuracy_score(y_true[idx], y_pred[idx])

    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return point, lo, hi


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model_name: ModelName,
    feature_name: str,
    save: bool = True,
) -> TrainedArtifact:
    """Fit one classifier on one feature representation and evaluate it.

    Args:
        X: feature matrix of shape (n_samples, n_features).
        y: integer labels of shape (n_samples,).
        model_name: which classifier from models.py to use.
        feature_name: tag identifying the feature representation ("mfcc",
            "wav2vec2", "yamnet"). Only used for saving artifacts.
        save: whether to write the trained pipeline and metrics to disk.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    pipe = build_classifier(model_name)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)

    # Headline metrics with bootstrap CI on accuracy.
    acc, acc_lo, acc_hi = _bootstrap_accuracy_ci(y_test, y_pred)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(EMOTION_LABELS))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(EMOTION_LABELS))),
        target_names=EMOTION_LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics: dict[str, Any] = {
        "feature_name": feature_name,
        "model_name": model_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": acc,
        "accuracy_ci_lo": acc_lo,
        "accuracy_ci_hi": acc_hi,
        "accuracy_ci_level": BOOTSTRAP_CI,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    artifact = TrainedArtifact(
        model_name=model_name,
        feature_name=feature_name,
        pipeline=pipe,
        X_test=X_test,
        y_test=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        metrics=metrics,
    )

    if save:
        _persist(artifact)

    return artifact


def _persist(artifact: TrainedArtifact) -> None:
    """Write the metrics JSON and pickled pipeline for one trained model."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = METRICS_DIR / f"{artifact.key}.json"
    with metrics_path.open("w") as f:
        json.dump(artifact.metrics, f, indent=2)

    model_path = METRICS_DIR / f"{artifact.key}.joblib"
    joblib.dump(artifact.pipeline, model_path)
