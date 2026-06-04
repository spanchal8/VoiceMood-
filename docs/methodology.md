# Methodology

This document describes VoiceMood's experimental protocol — the choices
made for data splitting, training, and evaluation, along with the reasoning
behind each.

## 1. Dataset

We use the **Ryerson Audio-Visual Database of Emotional Speech and Song
(RAVDESS)**, audio-only speech subset:

- 24 professional actors (12 female, 12 male)
- 8 emotion categories: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- 1,440 utterances total (60 per actor)
- Two intensity levels (normal, strong) per emotion
- Two lexically-matched statements
- All recordings: 16-bit, 48 kHz, mono (we resample to 16 kHz)

RAVDESS was chosen because:
- It is **acted but controlled**: every speaker produces every emotion, so
  the same speaker's voice appears in every class — class balance is exact.
- It is **small enough** to enable careful per-clip analysis (~1,440 samples).
- It is **standard** in the speech emotion recognition literature, so
  results are directly comparable to published numbers.
- It is **free**, no login required (Zenodo record 1188976).

### Known limitations
- Acted emotion is not the same as naturally-occurring emotion.
- North American English speakers only.
- Two short utterances ("Kids are talking by the door"; "Dogs are sitting
  by the door"). Lexical content does not carry emotion.

## 2. Feature representations

We compare three feature extractors:

### 2.1 Classical (MFCC + acoustic statistics)

A 170-dimensional vector per clip composed of mean and standard deviation
over time of:
- 40 MFCC coefficients (80 dims)
- 40 delta-MFCC (first temporal derivative) (80 dims)
- Zero-crossing rate (2 dims)
- Spectral centroid, bandwidth, rolloff (6 dims)
- RMS energy (2 dims)

These features are computed locally on the laptop in librosa — milliseconds
per clip. No pretrained model required.

### 2.2 wav2vec2 embeddings

`facebook/wav2vec2-base`, a self-supervised speech representation model
pretrained on 960 hours of Librispeech. We take the last hidden state
(time × 768), mean-pool across time to get a single 768-d vector per clip.

The model is frozen — we only use it as a feature extractor. This is the
**linear probe** protocol, which is the standard way to evaluate the
quality of self-supervised representations.

### 2.3 YAMNet embeddings

Google's YAMNet, a MobileNet-V1-based audio event classifier pretrained on
AudioSet. We take the penultimate embedding layer (time × 1024), mean-pool
across time. Same linear-probe protocol.

YAMNet is included because:
- It hits the TensorFlow requirement in the JD.
- It was trained on general audio events (not just speech), giving a
  qualitatively different representation.
- It is small and fast — a useful baseline.

## 3. Classifiers

For each feature set, we train three sklearn classifiers:

| Classifier | Why |
|---|---|
| Logistic regression (L2-regularised) | Linear, interpretable, fast baseline |
| Random forest (300 trees) | Non-linear, handles feature interactions |
| SVM with RBF kernel | Classic strong baseline on small datasets |

All three are wrapped in a `Pipeline` that standardises features
(zero mean, unit variance) before classification.

## 4. Train/test split

**Stratified random split**, 80% train / 20% test, seed fixed at 42.

The split is **stratified by emotion label** to ensure each emotion is
represented in both train and test sets in roughly the same proportion.

### Why not actor-stratified or leave-one-actor-out?

In the literature, leave-one-actor-out is sometimes preferred to test
speaker-independence. We use random stratification here because:

1. With 24 actors, leave-one-out would mean 24 separate training runs,
   inflating compute time substantially.
2. Random stratification matches the deployment scenario where the
   system has heard the user before.
3. The honest reporting of this choice in this document, plus the use
   of the same split across all 9 (feature × model) combinations,
   makes the comparison fair *within* this experimental design.

A speaker-independent version of this evaluation is left as future work,
explicitly called out in the README.

## 5. Metrics

### 5.1 Headline metrics

- **Accuracy** — fraction of test clips correctly classified.
- **Macro-averaged F1** — equally weights all 8 classes (robust to any
  residual class imbalance).
- **Multi-class Brier score** — calibration quality:
  $$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{i,c} - y_{i,c})^2$$

### 5.2 Confidence intervals

We report **95% bootstrap percentile confidence intervals** for accuracy.

Procedure: from the test set predictions, draw 1,000 bootstrap resamples
with replacement, compute accuracy on each, take the 2.5th and 97.5th
percentiles.

This is the right tool because:
- It makes no parametric assumptions about the accuracy distribution.
- The bottleneck (model training) is not re-run, so it's cheap.
- It directly addresses the question "how much would this accuracy
  number change with a different test draw?".

### 5.3 Paired classifier comparison: McNemar's test

To compare two classifiers on the same test set, we use McNemar's test
(Quinn McNemar, 1947), which is the appropriate test for **paired binary
outcomes** (here: each classifier's prediction is either correct or wrong
on the same input).

Build the 2 × 2 contingency table of (A correct, B correct), (A correct,
B wrong), (A wrong, B correct), (A wrong, B wrong). The test statistic is
based on the off-diagonal cells:

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

(With Yates's continuity correction, which is standard for small cells.)
Under the null hypothesis that both classifiers have the same error rate,
this is approximately chi-squared distributed with 1 degree of freedom.

We report the contingency table, the statistic, and the p-value for every
pair of classifiers within the same feature group (test sets must match
exactly for pairing to be valid).

### 5.4 Calibration: reliability diagrams

For a multi-class problem we build per-class one-vs-rest reliability
diagrams: for each class c, take the predicted probability p(c|x) and
the binary indicator 1[y = c]. Bin predictions into 10 equal-width bins
and plot the bin mean predicted probability against the bin empirical
positive rate. A perfectly calibrated classifier lies on the diagonal.

We average the curves across classes to produce one curve per model.
Combined with the Brier score, this tells us not just *how often* the
model is right, but whether its confidence is trustworthy.

## 6. Time-series analysis

RAVDESS clips are independent utterances, not conversations. To enable
time-series analysis we synthesise multi-clip conversations:

1. Group clips by actor.
2. For each actor, randomly partition their clips into non-overlapping
   chunks of 6.
3. Each chunk is one "conversation": a sequence of 6 emotion-probability
   vectors.

We then detect emotion change-points using a simple rule: a change is
recorded when the argmax-emotion label switches AND the new label
persists for at least 2 consecutive clips with mean predicted
probability ≥ 0.4. This filters out single-frame jitter.

This is intentionally a simple detector — the goal is to demonstrate the
analysis pattern (sequence of model outputs → temporal signal) rather
than to advance change-point detection theory.

## 7. Reproducibility

- Single random seed (42) used everywhere.
- Identical train/test split across all (feature × model) combinations
  within a feature group, so McNemar pairing is exact.
- All pinned dependency versions in `requirements.txt`.
- Bootstrap and synthetic-conversation seeds also fixed.

Running `python scripts/train_all.py` from a clean checkout reproduces
the results byte-for-byte (modulo BLAS-level non-determinism in some
sklearn solvers, which is too small to affect the reported numbers).
