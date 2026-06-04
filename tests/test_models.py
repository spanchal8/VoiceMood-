"""Smoke tests for the sklearn classifier factory."""

from __future__ import annotations

import numpy as np
import pytest

from voicemood.models import all_model_names, build_classifier


@pytest.mark.parametrize("name", all_model_names())
def test_classifier_fits_and_predicts(name) -> None:
    """Every classifier should fit on toy data and return valid probabilities."""
    rng = np.random.default_rng(0)
    n, d, k = 80, 12, 3
    X = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.integers(0, k, size=n)

    pipe = build_classifier(name)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)

    assert proba.shape == (n, k)
    # Each row is a valid probability distribution.
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert proba.min() >= 0.0 and proba.max() <= 1.0


def test_unknown_classifier_raises() -> None:
    with pytest.raises(ValueError):
        build_classifier("xgboost")  # type: ignore[arg-type]
