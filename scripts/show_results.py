"""Pretty-print the results from results/metrics/ as a terminal table.

Quick sanity check after train_all.py:
    python scripts/show_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from voicemood.config import METRICS_DIR  # noqa: E402


def main() -> int:
    files = sorted(METRICS_DIR.glob("*.json"))
    # Skip mcnemar (printed separately).
    metric_files = [f for f in files if not f.name.startswith("mcnemar")]
    if not metric_files:
        print(f"No metric files found in {METRICS_DIR}.")
        print("Run `python scripts/train_all.py` first.")
        return 1

    print()
    print(f"{'Feature':<12} {'Model':<22} {'Accuracy':<10} {'95% CI':<22} {'Macro F1':<10}")
    print("-" * 80)
    rows = []
    for fp in metric_files:
        with fp.open() as f:
            m = json.load(f)
        rows.append((
            m["feature_name"],
            m["model_name"],
            m["accuracy"],
            (m["accuracy_ci_lo"], m["accuracy_ci_hi"]),
            m["macro_f1"],
        ))
    # Sort by accuracy descending.
    rows.sort(key=lambda r: -r[2])
    for feat, model, acc, ci, f1 in rows:
        ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        print(f"{feat:<12} {model:<22} {acc:<10.3f} {ci_str:<22} {f1:<10.3f}")
    print()

    mc_path = METRICS_DIR / "mcnemar_pairs.json"
    if mc_path.exists():
        with mc_path.open() as f:
            pairs = json.load(f)
        sig = [p for p in pairs if p["p_value"] < 0.05]
        print(f"McNemar pairs computed: {len(pairs)}, statistically significant (p<0.05): {len(sig)}")
        if sig:
            print("  Significant disagreements:")
            for p in sig:
                print(f"    {p['model_a']} vs {p['model_b']}: chi²={p['chi2']:.2f}, p={p['p_value']:.4f}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
