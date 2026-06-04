"""Smoke tests for MFCC feature extraction."""

from __future__ import annotations

import numpy as np

from voicemood.features_mfcc import extract_mfcc_features, feature_dim
from voicemood.config import SAMPLE_RATE


def test_feature_dim_matches_extractor() -> None:
    """The advertised feature dim should match what the extractor returns."""
    sr = SAMPLE_RATE
    # 2 seconds of synthetic audio.
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(sr * 2).astype(np.float32)
    feats = extract_mfcc_features(audio, sr=sr)
    assert feats.ndim == 1
    assert feats.shape[0] == feature_dim()


def test_features_are_finite() -> None:
    """No NaNs or infs in features extracted from a clean tone."""
    sr = SAMPLE_RATE
    t = np.arange(sr * 1) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    feats = extract_mfcc_features(tone, sr=sr)
    assert np.all(np.isfinite(feats))


def test_features_change_with_different_audio() -> None:
    """Different audio should give measurably different features."""
    sr = SAMPLE_RATE
    rng = np.random.default_rng(0)
    a = rng.standard_normal(sr).astype(np.float32)
    b = (0.5 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype(np.float32)
    fa = extract_mfcc_features(a, sr=sr)
    fb = extract_mfcc_features(b, sr=sr)
    # They should not be coincidentally equal.
    assert not np.allclose(fa, fb)
