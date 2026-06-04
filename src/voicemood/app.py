"""Streamlit interactive demo for VoiceMood.

Design notes:
    - Loads ONLY the MFCC-based sklearn pipeline. This is the model that
      doesn't require any precomputed deep embeddings at inference time.
    - Loading is instant (< 100 ms). No deep model anywhere in this app.
    - Accepts uploaded .wav files OR live microphone recordings (where
      the browser allows it).
    - Renders waveform, prediction, full probability bar chart, and a
      short methodology note.
"""

from __future__ import annotations

import io
from pathlib import Path

import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from voicemood.config import EMOTION_LABELS, METRICS_DIR, SAMPLE_RATE
from voicemood.features_mfcc import extract_mfcc_features

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="VoiceMood — Speech Emotion Recognition",
    page_icon="🎙️",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    """Load the MFCC pipeline once and cache it for the session.

    We prefer the best-performing MFCC-based model; if none of the
    expected files exist, we surface a clear error.
    """
    candidates = [
        METRICS_DIR / "mfcc__svm_rbf.joblib",
        METRICS_DIR / "mfcc__random_forest.joblib",
        METRICS_DIR / "mfcc__logistic_regression.joblib",
    ]
    for p in candidates:
        if p.exists():
            return joblib.load(p), p.name
    return None, None


def _predict(audio: np.ndarray, pipeline) -> tuple[str, np.ndarray]:
    """Run a single-clip prediction. Returns (label, prob_vector)."""
    feats = extract_mfcc_features(audio).reshape(1, -1)
    probs = pipeline.predict_proba(feats)[0]
    label = EMOTION_LABELS[int(np.argmax(probs))]
    return label, probs


def _plot_waveform(audio: np.ndarray, sr: int):
    fig, ax = plt.subplots(figsize=(8, 2.5))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig


def _plot_mel(audio: np.ndarray, sr: int):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(
        mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("Mel spectrogram")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🎙️ VoiceMood")
st.caption(
    "Speech emotion recognition with MFCC features + scikit-learn classifier. "
    "Trained on RAVDESS. Runs entirely on your laptop CPU."
)

pipeline, pipeline_filename = _load_pipeline()

if pipeline is None:
    st.error(
        "No trained MFCC model found in `results/metrics/`. "
        "Run `python scripts/train_all.py` first."
    )
    st.stop()

with st.sidebar:
    st.header("Model")
    st.write(f"**Loaded:** `{pipeline_filename}`")
    st.write(f"**Sample rate:** {SAMPLE_RATE} Hz (mono)")
    st.write(f"**Classes:** {len(EMOTION_LABELS)}")
    st.markdown("---")
    st.markdown(
        "**How it works**\n\n"
        "1. Your audio is resampled to 16 kHz mono.\n"
        "2. We extract 170 acoustic features (MFCCs, deltas, spectral "
        "statistics).\n"
        "3. A scikit-learn classifier predicts the emotion with calibrated "
        "probabilities.\n\n"
        "No cloud calls. Everything runs locally."
    )

uploaded = st.file_uploader(
    "Upload a `.wav` clip (3–5 seconds of speech works best)",
    type=["wav", "mp3", "flac", "ogg"],
    help="Use a clip with one speaker and minimal background noise.",
)

if uploaded is None:
    st.info("Upload an audio file to get a prediction.")
    st.stop()

# Load & resample.
audio_bytes = uploaded.read()
try:
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE, mono=True)
except Exception as exc:  # pragma: no cover
    st.error(f"Could not decode audio: {exc}")
    st.stop()

if len(audio) < SAMPLE_RATE // 4:
    st.warning("Clip is very short (<0.25 s). Predictions may be unreliable.")

# Audio playback.
st.audio(audio_bytes)

# Prediction.
label, probs = _predict(audio, pipeline)

col_pred, col_plot = st.columns([1, 2])

with col_pred:
    st.subheader("Prediction")
    st.metric(label="Predicted emotion", value=label.capitalize())
    st.metric(label="Confidence", value=f"{probs.max() * 100:.1f}%")

with col_plot:
    proba_df = (
        pd.DataFrame({"emotion": EMOTION_LABELS, "probability": probs})
        .sort_values("probability", ascending=True)
    )
    st.bar_chart(proba_df.set_index("emotion"))

st.markdown("---")
st.subheader("Audio inspection")

st.pyplot(_plot_waveform(audio, SAMPLE_RATE))
st.pyplot(_plot_mel(audio, SAMPLE_RATE))

with st.expander("What does the model actually see?"):
    st.write(
        "The classifier sees a 170-dimensional vector summarising the clip:"
    )
    st.code(
        "  - MFCC means and std (80 dims)\n"
        "  - Delta-MFCC means and std (80 dims)\n"
        "  - Zero crossing rate mean+std (2 dims)\n"
        "  - Spectral centroid mean+std (2 dims)\n"
        "  - Spectral bandwidth mean+std (2 dims)\n"
        "  - Spectral rolloff mean+std (2 dims)\n"
        "  - RMS energy mean+std (2 dims)",
        language="text",
    )
