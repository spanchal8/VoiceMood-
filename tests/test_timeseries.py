"""Tests for time-series emotion change detection."""

from __future__ import annotations

import numpy as np

from voicemood.config import NUM_CLASSES
from voicemood.timeseries import (
    detect_emotion_changes,
    predict_sequence,
    predictive_entropy,
)


def _make_seq(label_seq: list[int], confidence: float = 0.8) -> np.ndarray:
    """Build a (T, NUM_CLASSES) probability sequence from a label list."""
    T = len(label_seq)
    p = np.full((T, NUM_CLASSES), (1.0 - confidence) / (NUM_CLASSES - 1))
    for t, lab in enumerate(label_seq):
        p[t, lab] = confidence
    # Normalise just to be safe.
    p /= p.sum(axis=1, keepdims=True)
    return p


def test_predict_sequence_returns_argmax() -> None:
    seq = _make_seq([0, 1, 2, 0])
    pred = predict_sequence(seq)
    assert list(pred) == [0, 1, 2, 0]


def test_no_changes_when_label_constant() -> None:
    seq = _make_seq([3] * 10)
    changes = detect_emotion_changes(seq)
    assert changes == []


def test_detects_clear_change() -> None:
    # 4 of label 0, then 4 of label 5 — should detect one change.
    seq = _make_seq([0, 0, 0, 0, 5, 5, 5, 5])
    changes = detect_emotion_changes(seq, min_run_length=2, confidence_threshold=0.5)
    assert len(changes) == 1
    assert changes[0].index == 4


def test_ignores_single_frame_jitter() -> None:
    # One frame of label 7 inside a long run of 2.
    seq = _make_seq([2, 2, 2, 7, 2, 2, 2, 2])
    changes = detect_emotion_changes(seq, min_run_length=2, confidence_threshold=0.5)
    # Single-frame switch shouldn't be reported.
    assert all(c.to_label != "disgust" for c in changes)


def test_entropy_is_nonnegative() -> None:
    seq = _make_seq([0, 1, 2, 3])
    ent = predictive_entropy(seq)
    assert (ent >= 0).all()
