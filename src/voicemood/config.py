"""Central configuration: paths, constants, label mappings.

Everything that's "wired in" lives here so that the rest of the code is
parameter-free and easy to read.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---------------------------------------------------------------

# Project root resolves relative to this file's location, so things work
# regardless of where you run scripts from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"                  # RAVDESS .wav files
EMBED_DIR: Path = DATA_DIR / "embeddings"          # features.npz from Colab

RESULTS_DIR: Path = PROJECT_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
PLOTS_DIR: Path = RESULTS_DIR / "plots"

# --- Emotion labels (RAVDESS) -------------------------------------------

# RAVDESS encodes emotion in the 3rd token of each filename:
# Filename example: 03-01-06-01-02-01-12.wav
#  - 03 = modality (03 = audio-only)
#  - 01 = vocal channel (01 = speech)
#  - 06 = emotion       <-- this is the label
#  - 01 = intensity
#  - 02 = statement
#  - 01 = repetition
#  - 12 = actor ID
EMOTION_CODE_TO_LABEL: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

EMOTION_LABELS: list[str] = list(EMOTION_CODE_TO_LABEL.values())
LABEL_TO_INDEX: dict[str, int] = {lab: i for i, lab in enumerate(EMOTION_LABELS)}
INDEX_TO_LABEL: dict[int, str] = {i: lab for lab, i in LABEL_TO_INDEX.items()}

NUM_CLASSES: int = len(EMOTION_LABELS)

# --- Audio settings -----------------------------------------------------

SAMPLE_RATE: int = 16_000     # standard for wav2vec2 and YAMNet
N_MFCC: int = 40              # number of MFCC coefficients
HOP_LENGTH: int = 512
N_FFT: int = 1024

# --- Training settings --------------------------------------------------

RANDOM_SEED: int = 42
N_FOLDS: int = 5              # for cross-validation on small dataset
TEST_SIZE: float = 0.2

# Bootstrap settings for confidence intervals on accuracy
BOOTSTRAP_RESAMPLES: int = 1_000
BOOTSTRAP_CI: float = 0.95


def ensure_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (RAW_DIR, EMBED_DIR, METRICS_DIR, PLOTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
