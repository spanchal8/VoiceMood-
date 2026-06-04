"""Tests for statistical evaluation utilities."""

from __future__ import annotations

import numpy as np

from voicemood.evaluate import brier_score_multiclass, mcnemar_test


def test_mcnemar_identical_predictions() -> None:
    """When two classifiers agree everywhere, McNemar returns p=1."""
    y = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    pred = np.array([0, 1, 2, 1, 0, 2, 1, 0])
    res = mcnemar_test(y, pred, pred)
    assert res["b"] == 0 and res["c"] == 0
    assert res["p_value"] == 1.0


def test_mcnemar_obvious_disagreement() -> None:
    """When A is right and B is wrong on most disagreements, p should be small."""
    n = 200
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, size=n)
    # A: gets 95% right.
    pred_a = y.copy()
    flip_a = rng.choice(n, size=int(0.05 * n), replace=False)
    pred_a[flip_a] = (pred_a[flip_a] + 1) % 3
    # B: gets 60% right.
    pred_b = y.copy()
    flip_b = rng.choice(n, size=int(0.40 * n), replace=False)
    pred_b[flip_b] = (pred_b[flip_b] + 1) % 3

    res = mcnemar_test(y, pred_a, pred_b)
    # A should disagree productively far more often than B.
    assert res["b"] > res["c"]
    assert res["p_value"] < 0.01


def test_brier_score_perfect() -> None:
    """A perfect, fully-confident classifier scores 0."""
    y = np.array([0, 1, 2])
    proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    assert brier_score_multiclass(y, proba) == 0.0


def test_brier_score_uniform() -> None:
    """A uniform classifier on K classes scores 1 - 1/K."""
    K = 4
    n = 50
    y = np.zeros(n, dtype=int)
    proba = np.full((n, K), 1.0 / K)
    expected = 1.0 - 1.0 / K
    # The exact identity: mean over rows of sum_c (1/K - one_hot_c)^2
    #   = (K-1)*(1/K)^2 + (1 - 1/K)^2
    #   = (K-1)/K^2 + (K-1)^2/K^2
    #   = (K-1)(K)/K^2 = (K-1)/K
    assert abs(brier_score_multiclass(y, proba) - expected) < 1e-9
