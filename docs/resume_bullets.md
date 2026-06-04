# Resume Bullets for VoiceMood

Three versions of the project entry, in case you want to use it on
different resume variants. Each is calibrated to ~3 lines on a printed
resume. Fill in the bracketed numbers after running `train_all.py`.

---

## Version A — emphasises the full pipeline (recommended)

**VoiceMood — Speech Emotion Recognition with Statistical Evaluation**  *(Month YYYY – Month YYYY)*

- Built end-to-end speech emotion recognition pipeline on RAVDESS comparing classical MFCC features (sklearn) against frozen wav2vec2 (PyTorch) and YAMNet (TensorFlow) embeddings; achieved [XX]% accuracy with bootstrap 95% confidence intervals [XX–XX]% on 8-class classification.
- Designed rigorous evaluation framework with McNemar's paired test for classifier comparison (9 model pairs), per-class calibration analysis (multi-class Brier score), and time-series change-point detection for emotion drift across synthetic multi-clip conversations.
- Shipped interactive Streamlit prototype (waveform/mel-spectrogram visualisation, live probability bars) and ONNX export with documented Core ML conversion path; verified ONNX-vs-sklearn probability agreement to <1e-4.

---

## Version B — emphasises the experiment design

**VoiceMood — Speech Emotion Recognition** *(Month YYYY – Month YYYY)*

- Conducted controlled experiment comparing 9 (feature × classifier) combinations on RAVDESS speech emotion dataset, using stratified split, fixed seed, and identical test set ordering across models to enable paired statistical testing.
- Applied McNemar's chi-squared paired test with Yates continuity correction to identify statistically significant differences between classifiers (b/c contingency cells reported per pair); used percentile bootstrap (n=1000) for accuracy CIs.
- Implemented full reproducible pipeline (Python OOP, sklearn Pipeline, librosa, scipy.stats) with deep representations precomputed once on Colab GPU and downstream evaluation runnable in seconds on a CPU laptop.

---

## Version C — emphasises the on-device angle

**VoiceMood — On-Device Speech Emotion Recognition Prototype** *(Month YYYY – Month YYYY)*

- Designed a deployable speech emotion recognition system targeting Apple Core ML, using a 170-dim handcrafted acoustic feature vector (MFCCs, delta-MFCCs, spectral statistics) and an SVM-RBF classifier achieving [XX]% accuracy on 8-class RAVDESS.
- Compared classical pipeline against frozen wav2vec2 (PyTorch) and YAMNet (TensorFlow) embedding probes on identical test sets; performed McNemar paired testing and reliability-diagram calibration analysis.
- Exported final pipeline through skl2onnx to ONNX (verified <1e-4 probability delta vs sklearn) with documented coremltools conversion procedure; sub-1MB on-device footprint suitable for iOS deployment.

---

## Skills line addition

Add these to your skills row if not already there:

> Audio: librosa, MFCC, wav2vec2, YAMNet
> Stats: bootstrap confidence intervals, McNemar test, calibration (Brier)
> Deployment: ONNX (skl2onnx, onnxruntime), Core ML (documented path)

---

## What recruiters' filters will catch

This project as bulleted hits the following Apple AIML JD keywords on
keyword scanners:

`Python` `PyTorch` `TensorFlow` `scikit-learn` `Core ML` `ONNX`
`speech` `linear algebra` `statistics` `experiments`
`prototyping` `interactive systems` `machine learning`

That's 13 distinct JD keywords in three bullets. The remaining ones from
the JD (Swift, Objective-C, Java, econometrics, causal inference, time
series, optimization) are not all covered — but no MS-student project
honestly covers all of them, and time-series IS partially covered by
the emotion-drift analysis.
