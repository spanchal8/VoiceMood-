"""Train all three pipelines (MFCC, wav2vec2, YAMNet) on RAVDESS.

This is the main "do everything on the laptop" entry point.

Pipeline:
    1. Discover RAVDESS .wav files in data/raw/.
    2. Extract MFCC features locally (fast, ~10 ms per clip).
    3. Load wav2vec2 / YAMNet embeddings from data/embeddings/features.npz
       (precomputed once in Colab — see notebooks/).
    4. For each feature set: train logistic regression, random forest, and
       SVM-RBF with a stratified train/test split.
    5. Run pairwise McNemar tests across all 9 (feature x model) pairs.
    6. Save metrics, calibration plots, and confusion matrices.
    7. Write results/REPORT.md with a summary table.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --skip-deep    # MFCC only, no Colab needed

The --skip-deep flag is for the "I haven't run the Colab notebook yet"
case — you can still get a complete MFCC-only result in under 30 seconds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from voicemood.config import (  # noqa: E402
    EMBED_DIR,
    EMOTION_LABELS,
    LABEL_TO_INDEX,
    METRICS_DIR,
    PLOTS_DIR,
    RAW_DIR,
    RESULTS_DIR,
    ensure_dirs,
)
from voicemood.dataset import discover_clips  # noqa: E402
from voicemood.evaluate import (  # noqa: E402
    brier_score_multiclass,
    pairwise_mcnemar,
    plot_calibration,
    plot_confusion_matrix,
    save_mcnemar_table,
)
from voicemood.features_mfcc import extract_from_path  # noqa: E402
from voicemood.models import all_model_names  # noqa: E402
from voicemood.train import TrainedArtifact, train_and_evaluate  # noqa: E402


def _build_mfcc_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the (X_mfcc, y, actor_ids) matrices from local audio files."""
    clips = discover_clips(RAW_DIR)
    if not clips:
        raise FileNotFoundError(
            f"No RAVDESS clips found in {RAW_DIR}. "
            "Run `python scripts/download_ravdess.py` first."
        )

    X_rows: list[np.ndarray] = []
    y_rows: list[int] = []
    actors: list[int] = []
    for clip in tqdm(clips, desc="Extracting MFCC features"):
        try:
            feats = extract_from_path(clip.path)
        except Exception as exc:
            print(f"  skipping {clip.path.name}: {exc}", file=sys.stderr)
            continue
        X_rows.append(feats)
        y_rows.append(LABEL_TO_INDEX[clip.emotion_label])
        actors.append(clip.actor_id)

    return (
        np.stack(X_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.int64),
        np.asarray(actors, dtype=np.int64),
    )


def _load_deep_embeddings() -> dict[str, np.ndarray] | None:
    """Load the Colab-precomputed embeddings if present.

    Expected file: data/embeddings/features.npz with keys:
        wav2vec2 : (N, D_w2v) float32
        yamnet   : (N, D_yam) float32
        labels   : (N,)        int64
        actors   : (N,)        int64

    The labels and actors arrays should match the order of clips returned
    by discover_clips() — the Colab notebook follows the same convention.
    """
    path = EMBED_DIR / "features.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def main() -> int:
    parser = argparse.ArgumentParser(description="Train all VoiceMood pipelines")
    parser.add_argument(
        "--skip-deep",
        action="store_true",
        help="Skip wav2vec2/YAMNet pipelines (no Colab needed).",
    )
    args = parser.parse_args()

    ensure_dirs()
    t0 = time.perf_counter()

    # --- MFCC (laptop) ---------------------------------------------------
    X_mfcc, y, actors = _build_mfcc_matrix()
    print(f"  MFCC matrix: {X_mfcc.shape}, labels: {y.shape}")

    # --- Deep embeddings (precomputed in Colab) --------------------------
    deep = None if args.skip_deep else _load_deep_embeddings()
    if deep is None and not args.skip_deep:
        print(
            "\nNOTE: no precomputed embeddings found at "
            f"{EMBED_DIR / 'features.npz'}.\n"
            "      Running MFCC-only. Open the Colab notebook to add deep models.\n"
        )

    feature_sets: list[tuple[str, np.ndarray, np.ndarray]] = [("mfcc", X_mfcc, y)]
    if deep is not None:
        # Align with our local label order via the labels saved in the npz.
        # If actor ordering matches, we use the local y/actors; if not, we use
        # the npz's own labels (safer).
        deep_labels = deep["labels"].astype(np.int64)
        feature_sets.append(("wav2vec2", deep["wav2vec2"].astype(np.float32), deep_labels))
        feature_sets.append(("yamnet", deep["yamnet"].astype(np.float32), deep_labels))

    # --- Train every (feature, model) combination ------------------------
    artifacts: list[TrainedArtifact] = []
    for fname, X, y_used in feature_sets:
        for mname in all_model_names():
            print(f"\n  Training {fname} + {mname} ...")
            art = train_and_evaluate(
                X, y_used, model_name=mname, feature_name=fname, save=True
            )
            print(
                f"    accuracy={art.metrics['accuracy']:.3f} "
                f"(95% CI {art.metrics['accuracy_ci_lo']:.3f}"
                f"–{art.metrics['accuracy_ci_hi']:.3f}), "
                f"macro F1={art.metrics['macro_f1']:.3f}"
            )
            artifacts.append(art)

    # --- Statistical comparison ------------------------------------------
    # McNemar requires identical test sets. We grouped artifacts by feature
    # set above; train_and_evaluate uses the same seed everywhere, so test
    # sets are identical within each feature group. We compare *across*
    # models within each feature group.
    all_pairs: list[dict] = []
    by_feature: dict[str, list[TrainedArtifact]] = {}
    for art in artifacts:
        by_feature.setdefault(art.feature_name, []).append(art)
    for fname, group in by_feature.items():
        pairs = pairwise_mcnemar(group)
        for p in pairs:
            p["feature_group"] = fname
        all_pairs.extend(pairs)
    save_mcnemar_table(all_pairs)
    print(f"\n  McNemar pairs computed: {len(all_pairs)}")

    # --- Plots ------------------------------------------------------------
    for art in artifacts:
        plot_confusion_matrix(art)
        plot_calibration(art)

    # --- Summary report ---------------------------------------------------
    report_lines = ["# VoiceMood Results\n"]
    report_lines.append("## Headline metrics\n")
    report_lines.append("| Feature | Model | Accuracy | 95% CI | Macro F1 | Brier |")
    report_lines.append("|---|---|---|---|---|---|")
    for art in artifacts:
        brier = brier_score_multiclass(art.y_test, art.y_proba)
        ci = (
            f"[{art.metrics['accuracy_ci_lo']:.3f}, "
            f"{art.metrics['accuracy_ci_hi']:.3f}]"
        )
        report_lines.append(
            f"| {art.feature_name} | {art.model_name} | "
            f"{art.metrics['accuracy']:.3f} | {ci} | "
            f"{art.metrics['macro_f1']:.3f} | {brier:.3f} |"
        )

    report_lines.append("\n## Pairwise McNemar tests (within feature group)\n")
    report_lines.append("| Group | Model A | Model B | b (A only) | c (B only) | chi² | p |")
    report_lines.append("|---|---|---|---|---|---|---|")
    for p in all_pairs:
        report_lines.append(
            f"| {p['feature_group']} | {p['model_a']} | {p['model_b']} | "
            f"{p['b']} | {p['c']} | {p['chi2']:.3f} | {p['p_value']:.4f} |"
        )

    report_lines.append("\n## Confusion matrices\n")
    for art in artifacts:
        report_lines.append(f"### {art.key}\n")
        report_lines.append(f"![confusion]({Path('plots') / f'confusion_{art.key}.png'})\n")

    report_lines.append("\n## Calibration\n")
    for art in artifacts:
        report_lines.append(f"### {art.key}\n")
        report_lines.append(f"![calibration]({Path('plots') / f'calibration_{art.key}.png'})\n")

    elapsed = time.perf_counter() - t0
    report_lines.append(f"\n_Generated in {elapsed:.1f} s._\n")

    report_path = RESULTS_DIR / "REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n  Wrote {report_path}")

    print(f"\nDone in {elapsed:.1f} s. Results in {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
