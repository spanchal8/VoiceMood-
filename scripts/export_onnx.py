"""Export the best trained sklearn pipeline to ONNX.

Pipeline:
    1. Read all metric JSONs from results/metrics/.
    2. Pick the highest-accuracy MFCC pipeline (the one that actually runs
       on the user's laptop without precomputed deep embeddings).
    3. Convert it to ONNX.
    4. Verify ONNX output matches sklearn output on a held-out batch.

Usage:
    python scripts/export_onnx.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from voicemood.config import METRICS_DIR, RAW_DIR, RESULTS_DIR  # noqa: E402
from voicemood.dataset import discover_clips  # noqa: E402
from voicemood.features_mfcc import extract_from_path, feature_dim  # noqa: E402
from voicemood.onnx_export import (  # noqa: E402
    export_pipeline_to_onnx,
    verify_onnx_matches_sklearn,
)


def _best_mfcc_metrics() -> dict | None:
    best = None
    for fp in METRICS_DIR.glob("mfcc__*.json"):
        with fp.open() as f:
            m = json.load(f)
        if best is None or m["accuracy"] > best["accuracy"]:
            best = m
    return best


def main() -> int:
    best = _best_mfcc_metrics()
    if best is None:
        print(f"No MFCC metric files in {METRICS_DIR}. Run train_all.py first.")
        return 1

    feature_name = best["feature_name"]
    model_name = best["model_name"]
    pipeline_path = METRICS_DIR / f"{feature_name}__{model_name}.joblib"
    onnx_path = RESULTS_DIR / "voicemood.onnx"

    print(f"Best MFCC model: {model_name} (accuracy = {best['accuracy']:.3f})")
    print(f"Exporting {pipeline_path.name} -> {onnx_path.name}")
    export_pipeline_to_onnx(
        pipeline_path,
        onnx_path,
        n_features=feature_dim(),
        model_name="voicemood",
    )
    print(f"  wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")

    # Verify with a fresh batch of features.
    clips = discover_clips(RAW_DIR)[:8]
    if not clips:
        print("  (no clips available for verification; skipping consistency check)")
        return 0

    X_sample = np.stack([extract_from_path(c.path) for c in clips]).astype(np.float32)
    result = verify_onnx_matches_sklearn(pipeline_path, onnx_path, X_sample)
    status = "PASS" if result["passes_tolerance"] else "FAIL"
    print(
        f"  ONNX vs sklearn consistency: {status} "
        f"(max abs diff = {result['max_abs_diff']:.2e})"
    )
    return 0 if result["passes_tolerance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
