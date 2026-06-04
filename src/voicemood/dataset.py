"""Utilities for parsing RAVDESS filenames and discovering audio files.

RAVDESS filename schema:
    <modality>-<vocal_channel>-<emotion>-<intensity>-<statement>-<repetition>-<actor>.wav

We only use the audio-only subset (modality == 03) and treat each .wav
as one sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import EMOTION_CODE_TO_LABEL, RAW_DIR


@dataclass(frozen=True)
class RavdessClip:
    """A single RAVDESS audio clip with parsed metadata."""

    path: Path
    actor_id: int            # 1..24
    emotion_code: str        # "01".."08"
    emotion_label: str       # "neutral" .. "surprised"
    intensity: int           # 1 normal, 2 strong
    statement: int           # 1 or 2
    repetition: int          # 1 or 2

    @property
    def gender(self) -> str:
        # RAVDESS: odd actor IDs are male, even are female
        return "male" if self.actor_id % 2 == 1 else "female"


def parse_ravdess_filename(path: Path) -> RavdessClip | None:
    """Parse a RAVDESS .wav filename into a RavdessClip.

    Returns None if the filename doesn't match the expected pattern or
    if the modality is not audio-only.
    """
    stem = path.stem
    parts = stem.split("-")
    if len(parts) != 7:
        return None

    modality, vocal_channel, emotion_code, intensity, statement, repetition, actor = parts

    # We only handle audio-only (03), speech (01)
    if modality != "03" or vocal_channel != "01":
        return None

    if emotion_code not in EMOTION_CODE_TO_LABEL:
        return None

    try:
        return RavdessClip(
            path=path,
            actor_id=int(actor),
            emotion_code=emotion_code,
            emotion_label=EMOTION_CODE_TO_LABEL[emotion_code],
            intensity=int(intensity),
            statement=int(statement),
            repetition=int(repetition),
        )
    except ValueError:
        return None


def discover_clips(root: Path | None = None) -> list[RavdessClip]:
    """Walk the RAVDESS directory and return all parseable clips.

    The RAVDESS zip extracts to Actor_01..Actor_24 subdirectories, but
    we recurse so the user can drop files in any layout.
    """
    root = root or RAW_DIR
    if not root.exists():
        return []

    clips: list[RavdessClip] = []
    for wav_path in sorted(root.rglob("*.wav")):
        clip = parse_ravdess_filename(wav_path)
        if clip is not None:
            clips.append(clip)
    return clips
