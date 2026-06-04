# VoiceMood — Speech Emotion Recognition with Statistical Evaluation

A speech emotion recognition system that compares classical machine learning
(MFCC + scikit-learn) against modern deep learning representations
(wav2vec2 embeddings), with rigorous statistical evaluation and an interactive
Streamlit prototype. Designed to be deployable on-device via ONNX with a
documented path to Apple Core ML.

---

## What it does

Given a short voice clip, VoiceMood predicts the speaker's emotional state
(neutral, calm, happy, sad, angry, fearful, disgust, surprised) and returns
calibrated confidence scores. It ships with:

- A classical baseline (handcrafted acoustic features + sklearn classifiers)
- A deep baseline (frozen wav2vec2 embeddings + linear probe)
- A TensorFlow baseline (YAMNet embeddings + linear probe)
- A rigorous evaluation framework with bootstrap confidence intervals,
  McNemar's test for paired classifier comparison, and calibration curves
- A time-series analysis of emotion dynamics across multi-clip "conversations"
- An interactive Streamlit prototype for live demos
- ONNX export with a documented Core ML conversion path

---

## Architecture at a glance

```
                  ┌────────────────────────────────────────────┐
                  │            ONE-TIME COLAB STEP             │
                  │  (free GPU, runs in browser, ~5 min total) │
                  ├────────────────────────────────────────────┤
                  │  RAVDESS dataset → wav2vec2 embeddings     │
                  │  RAVDESS dataset → YAMNet embeddings       │
                  │  Saves: features.npz (~10 MB)              │
                  └────────────────┬───────────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────────┐
              │       EVERYTHING ELSE ON YOUR LAPTOP         │
              │           (CPU, fast, no big models)         │
              ├──────────────────────────────────────────────┤
              │  features.npz                                │
              │       │                                      │
              │       ├──► MFCC pipeline (sklearn)           │
              │       ├──► wav2vec2 probe (sklearn)          │
              │       └──► YAMNet probe (sklearn)            │
              │                                              │
              │  Evaluation:                                 │
              │    - bootstrap CIs                           │
              │    - McNemar's paired test                   │
              │    - calibration curves                      │
              │    - time-series emotion drift               │
              │                                              │
              │  Streamlit demo + ONNX export                │
              └──────────────────────────────────────────────┘
```

The design rule: **the laptop never loads a deep learning model**.
All heavy lifting happens once in Colab, and the laptop only sees small
numpy arrays of precomputed embeddings. This keeps every interactive step
under one second.

---

## Quickstart (30 minutes, end-to-end)

### Prerequisites

- Python 3.10 or 3.11 on Windows / macOS / Linux
- A Google account (for the free Colab step)
- ~500 MB free disk space

### Step 1 — Install (laptop, ~2 min)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### Step 2 — Get the embeddings (Colab, ~5 min)

1. Open `notebooks/01_extract_embeddings_colab.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU → Save.
3. Runtime → Run all.
4. When it finishes, it downloads `features.npz` to your laptop automatically.
5. Move that file into `data/embeddings/`.

### Step 3 — Train everything (laptop, < 1 min)

```bash
python scripts/train_all.py
```

This trains the MFCC, wav2vec2, and YAMNet probes, runs the statistical
evaluation, and writes everything to `results/`.

### Step 4 — Look at the results

```bash
python scripts/show_results.py
```

Or open `results/REPORT.md` for the writeup with plots inline.

### Step 5 — Run the demo

```bash
streamlit run src/voicemood/app.py
```

Browser opens at `http://localhost:8501`. Upload a `.wav` file and see
the prediction, confidence breakdown, and waveform.

### Step 6 — Export to ONNX

```bash
python scripts/export_onnx.py
```

Writes `results/voicemood.onnx`. See `docs/coreml_path.md` for the
documented Core ML conversion procedure.

---

## Repository layout

```
voicemood/
├── README.md                          this file
├── requirements.txt                   pinned dependency versions
├── LICENSE
│
├── src/voicemood/
│   ├── __init__.py
│   ├── config.py                      paths, constants, emotion labels
│   ├── features_mfcc.py               classical MFCC + stats features
│   ├── models.py                      sklearn classifier factory
│   ├── train.py                       train one classifier given features
│   ├── evaluate.py                    bootstrap CIs, McNemar, calibration
│   ├── timeseries.py                  emotion-drift analysis
│   ├── onnx_export.py                 sklearn → ONNX
│   └── app.py                         Streamlit prototype
│
├── notebooks/
│   ├── 01_extract_embeddings_colab.ipynb   one-time heavy step
│   └── 02_results_exploration.ipynb        optional analysis
│
├── scripts/
│   ├── download_ravdess.py            laptop-side dataset download
│   ├── train_all.py                   trains all 3 pipelines
│   ├── show_results.py                pretty-prints results
│   └── export_onnx.py                 ONNX export entry point
│
├── data/
│   ├── raw/                           RAVDESS wav files (gitignored)
│   └── embeddings/                    features.npz from Colab
│
├── results/
│   ├── metrics/                       JSON per model
│   ├── plots/                         PNGs
│   └── REPORT.md                      auto-generated writeup
│
├── docs/
│   ├── methodology.md                 experimental protocol
│   ├── coreml_path.md                 ONNX → Core ML procedure
│   └── apple_interview_talking_points.md
│
└── tests/
    ├── test_features.py
    ├── test_models.py
    └── test_evaluate.py
```

---



## Honest scope notes

- The Streamlit demo classifies *uploaded* `.wav` files reliably. Live
  microphone recording works on most browsers but the audio capture
  permissions can be flaky on Windows; uploading a recorded file is the
  fallback path.
- The wav2vec2 and YAMNet embedding extraction is **only** done in Colab.
  The laptop never has to load these models. This is deliberate.
- ONNX export covers the sklearn classifiers. The deep models are not
  exported because they live upstream in Colab; their conversion path
  is documented in `docs/coreml_path.md` for completeness.
- All datasets, models, and APIs used here are free with no rate limits
  for the project's scale.

---

## License

MIT — see `LICENSE`.
