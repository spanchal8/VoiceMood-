"""Statistical evaluation across trained models.

This module is what makes the project defensible as research, not just code.
It computes:
    - McNemar's paired test for any two classifiers on the same test set
    - Reliability diagrams (calibration) with the Brier score
    - Confusion matrix heatmaps

McNemar's test (Quinn McNemar, 1947) is the *correct* test for comparing
two classifiers evaluated on the same test data, because the predictions
are paired (same input, two outputs). A standard chi-square or t-test
treats them as independent samples and inflates significance.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2
from sklearn.calibration import calibration_curve

from .config import EMOTION_LABELS, METRICS_DIR, NUM_CLASSES, PLOTS_DIR
from .train import TrainedArtifact

# ---------------------------------------------------------------------------
# McNemar's paired test
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    use_continuity: bool = True,
) -> dict[str, float]:
    """McNemar's test for paired classifier predictions.

    Build the 2x2 contingency table:

                          B correct   B wrong
        A correct  ->        a           b
        A wrong    ->        c           d

    The off-diagonal cells (b, c) are the disagreements. Under H0 (same
    error rate), b and c are exchangeable, so the statistic

        chi^2 = (|b - c| - 1)^2 / (b + c)         (with continuity correction)

    is approximately chi-squared with 1 df.

    Returns a dict with the table entries, the statistic, and the p-value.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # We only care about the disagreement cells.
    b = int(np.sum(correct_a & ~correct_b))     # A right, B wrong
    c = int(np.sum(~correct_a & correct_b))     # A wrong, B right
    a = int(np.sum(correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))

    if (b + c) == 0:
        # Identical predictions everywhere they matter.
        return {"a": a, "b": b, "c": c, "d": d, "chi2": 0.0, "p_value": 1.0}

    if use_continuity:
        stat = ((abs(b - c) - 1) ** 2) / (b + c)
    else:
        stat = ((b - c) ** 2) / (b + c)

    # 1 - CDF gives the upper-tail p-value for a chi^2 with 1 df.
    p_value = float(1.0 - chi2.cdf(stat, df=1))

    return {"a": a, "b": b, "c": c, "d": d, "chi2": float(stat), "p_value": p_value}


def pairwise_mcnemar(
    artifacts: Iterable[TrainedArtifact],
) -> list[dict]:
    """Run McNemar's test for every pair of artifacts (same test set assumed)."""
    arts = list(artifacts)
    out: list[dict] = []
    for a, b in combinations(arts, 2):
        # Sanity: artifacts must share the same y_test ordering.
        if not np.array_equal(a.y_test, b.y_test):
            raise ValueError(
                f"y_test mismatch between {a.key} and {b.key}; "
                "McNemar requires identical test set ordering."
            )
        result = mcnemar_test(a.y_test, a.y_pred, b.y_pred)
        result["model_a"] = a.key
        result["model_b"] = b.key
        out.append(result)
    return out


def save_mcnemar_table(results: list[dict], path: Path | None = None) -> Path:
    path = path or (METRICS_DIR / "mcnemar_pairs.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(results, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def brier_score_multiclass(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Multi-class Brier score: mean squared error against one-hot labels.

    A perfectly calibrated and confident classifier scores 0; a uniform
    classifier on K classes scores 1 - 1/K.
    """
    one_hot = np.eye(y_proba.shape[1])[y_true]
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def plot_calibration(artifact: TrainedArtifact, save: bool = True) -> Path | None:
    """One-vs-rest reliability diagram, averaged across classes.

    For each class c, take p(c|x) from y_proba and the binary indicator
    1[y_true == c], then compute a calibration curve. We average the
    curves across classes to get a single line per model.
    """
    n_bins = 10
    mean_fraction = np.zeros(n_bins)
    mean_pred = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for c in range(NUM_CLASSES):
        prob_c = artifact.y_proba[:, c]
        true_c = (artifact.y_test == c).astype(np.int64)
        try:
            frac_pos, mean_p = calibration_curve(
                true_c, prob_c, n_bins=n_bins, strategy="uniform"
            )
        except ValueError:
            # Happens when a class has no predictions in some bins.
            continue

        # Right-align partial curves into the bin grid.
        for fp, mp in zip(frac_pos, mean_p):
            bin_idx = min(int(mp * n_bins), n_bins - 1)
            mean_fraction[bin_idx] += fp
            mean_pred[bin_idx] += mp
            bin_counts[bin_idx] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_fraction = np.where(bin_counts > 0, mean_fraction / np.maximum(bin_counts, 1), np.nan)
        mean_pred = np.where(bin_counts > 0, mean_pred / np.maximum(bin_counts, 1), np.nan)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect calibration")
    ax.plot(mean_pred, mean_fraction, "o-", label=artifact.key)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration — {artifact.key}")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PLOTS_DIR / f"calibration_{artifact.key}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    plt.close(fig)
    return None


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(artifact: TrainedArtifact, save: bool = True) -> Path | None:
    """Heatmap of per-class predictions, row-normalised."""
    cm = np.array(artifact.metrics["confusion_matrix"], dtype=np.float64)
    # Row-normalise so each row shows the fraction predicted as each class.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS,
        cbar_kws={"label": "Row-normalised fraction"},
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {artifact.key}")
    fig.tight_layout()

    if save:
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        out = PLOTS_DIR / f"confusion_{artifact.key}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out
    plt.close(fig)
    return None
