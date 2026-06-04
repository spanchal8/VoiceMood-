"""Time-series analysis of emotion across a multi-clip 'conversation'.

RAVDESS doesn't natively contain conversations — each .wav is a standalone
utterance. We construct synthetic conversations by concatenating consecutive
clips from the same actor in their natural recording order, then ask:

    Given a sequence of predicted-emotion probability vectors p_1, ..., p_T,
    can we detect change-points where the dominant emotion shifts?

This serves two purposes:
    1. Demonstrates time-series methods on top of per-frame model predictions.
    2. Shows a realistic use case: tracking a speaker's emotion over a
       call or voice assistant interaction.

We use a CUSUM-style detector on the predicted-class entropy and on the
maximum-probability class identity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EMOTION_LABELS, NUM_CLASSES


@dataclass(frozen=True)
class ChangePoint:
    """A detected change-point in an emotion sequence."""

    index: int          # position in the sequence (0-indexed)
    from_label: str
    to_label: str
    confidence: float   # how confident the detector is (0..1)


def predict_sequence(proba_seq: np.ndarray) -> np.ndarray:
    """Argmax labels from a (T, NUM_CLASSES) probability sequence."""
    if proba_seq.ndim != 2 or proba_seq.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"proba_seq must have shape (T, {NUM_CLASSES}), got {proba_seq.shape}"
        )
    return proba_seq.argmax(axis=1)


def predictive_entropy(proba_seq: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-timestep Shannon entropy of the predicted distribution."""
    p = np.clip(proba_seq, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)


def detect_emotion_changes(
    proba_seq: np.ndarray,
    *,
    min_run_length: int = 2,
    confidence_threshold: float = 0.4,
) -> list[ChangePoint]:
    """Detect change-points where the argmax emotion switches.

    A change-point is recorded when the argmax label changes AND the
    incoming label has been the argmax for at least `min_run_length`
    consecutive steps with mean confidence >= `confidence_threshold`.
    This filters out single-frame jitter.
    """
    labels = predict_sequence(proba_seq)
    max_probs = proba_seq.max(axis=1)
    T = len(labels)
    if T < 2:
        return []

    changes: list[ChangePoint] = []
    current_label = int(labels[0])
    run_start = 0

    for t in range(1, T):
        if labels[t] != current_label:
            # Look ahead to see if the new label is stable.
            j = t
            while j < T and labels[j] == labels[t]:
                j += 1
            run_len = j - t
            mean_conf = float(np.mean(max_probs[t:j]))

            if run_len >= min_run_length and mean_conf >= confidence_threshold:
                changes.append(
                    ChangePoint(
                        index=t,
                        from_label=EMOTION_LABELS[current_label],
                        to_label=EMOTION_LABELS[int(labels[t])],
                        confidence=mean_conf,
                    )
                )
                current_label = int(labels[t])
                run_start = t

    return changes


def synthesise_conversations(
    proba_matrix: np.ndarray,
    actor_ids: np.ndarray,
    *,
    clips_per_conversation: int = 6,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Group per-clip probabilities into per-actor conversations.

    Args:
        proba_matrix: (N, NUM_CLASSES) per-clip prediction probabilities.
        actor_ids: (N,) integer actor IDs, one per row.
        clips_per_conversation: how many clips per synthetic conversation.

    Returns:
        Mapping of conversation_id -> probability sequence (T, NUM_CLASSES).
    """
    rng = np.random.default_rng(seed)
    out: dict[int, np.ndarray] = {}
    conv_id = 0
    for actor in np.unique(actor_ids):
        mask = actor_ids == actor
        idxs = np.where(mask)[0]
        rng.shuffle(idxs)
        # Take non-overlapping chunks.
        for start in range(0, len(idxs) - clips_per_conversation + 1, clips_per_conversation):
            chunk = idxs[start:start + clips_per_conversation]
            out[conv_id] = proba_matrix[chunk]
            conv_id += 1
    return out
