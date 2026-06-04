"""Classical handcrafted acoustic feature extraction.

This is the "explainable, fast, defendable" pipeline. Everything here runs
on a laptop CPU in milliseconds per clip. No deep models, no GPU, no big
downloads.

Features extracted (all per-clip, summarised across time):
    - MFCC mean + std (40 coefficients * 2 = 80 dims)
    - Delta-MFCC mean + std (80 dims)
    - Zero crossing rate mean + std (2 dims)
    - Spectral centroid mean + std (2 dims)
    - Spectral bandwidth mean + std (2 dims)
    - Spectral rolloff mean + std (2 dims)
    - RMS energy mean + std (2 dims)

Total: 170 features per clip. Small enough that sklearn classifiers train
in seconds.
"""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .config import HOP_LENGTH, N_FFT, N_MFCC, SAMPLE_RATE


def _summarise(arr: np.ndarray) -> np.ndarray:
    """Reduce a (n_features, n_frames) array to (2*n_features,) by mean+std."""
    return np.concatenate([arr.mean(axis=1), arr.std(axis=1)])


def extract_mfcc_features(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Extract a fixed-size feature vector from a 1D audio array.

    Args:
        audio: mono float32 PCM in [-1, 1].
        sr: sample rate (audio is resampled by the caller).

    Returns:
        1D float32 numpy array of length 170.
    """
    # MFCCs and their first temporal derivative.
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)

    # Spectral / temporal stats. Each returns (1, n_frames) or (n_frames,).
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=HOP_LENGTH)
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=HOP_LENGTH)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)

    feats = np.concatenate([
        _summarise(mfcc),
        _summarise(delta),
        _summarise(zcr),
        _summarise(centroid),
        _summarise(bandwidth),
        _summarise(rolloff),
        _summarise(rms),
    ]).astype(np.float32)

    return feats


def extract_from_path(path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a wav file, resample to target_sr, and extract features."""
    audio, _ = librosa.load(str(path), sr=target_sr, mono=True)
    return extract_mfcc_features(audio, sr=target_sr)


def feature_dim() -> int:
    """Return the expected feature vector length (handy for tests)."""
    # 2 * (N_MFCC + N_MFCC + 1 + 1 + 1 + 1 + 1)
    return 2 * (N_MFCC + N_MFCC + 5)
