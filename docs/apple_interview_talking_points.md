# Interview Talking Points

Quick reference for talking about VoiceMood in technical interviews and
recruiter screens. Each point is short, defendable, and connects back to
something concrete in the codebase.

## The one-sentence pitch

> "VoiceMood is a speech emotion recognition system that compares
> classical MFCC features against modern self-supervised speech
> representations, with rigorous statistical evaluation and a clear
> path to on-device Apple deployment."

## Why this project

> "I wanted to build something that hit three things at once: a real
> speech / multimodal problem, statistically careful evaluation, and an
> on-device deployment story. RAVDESS is small enough to do all of that
> in one project without cutting corners."

## Architecture decisions you should expect to defend

### Q: Why MFCC AND wav2vec2 AND YAMNet?
> "Different feature representations give different inductive biases.
> MFCCs are interpretable and fast — they encode short-time spectral
> envelope. wav2vec2 is self-supervised on speech, so it learns
> phonetic-level structure. YAMNet was trained on general audio events,
> so it captures broader acoustic patterns. Comparing all three on the
> same classifier lets me isolate what the *representation* contributes
> versus what the *classifier* contributes."

### Q: Why frozen embeddings, not fine-tuning?
> "Two reasons. First, linear probing is the standard way to evaluate
> representation quality — if I fine-tuned, I'd be measuring training
> tricks rather than the representation. Second, frozen embeddings work
> on a small dataset like RAVDESS without overfitting; fine-tuning a
> 95M-parameter model on 1,400 examples is asking for trouble."

### Q: Why McNemar's test?
> "When you compare two classifiers on the same test set, their
> predictions are paired — same input, two outputs. A regular chi-squared
> or t-test treats them as independent and inflates significance.
> McNemar's test is built specifically for paired binary outcomes."

### Q: Why bootstrap confidence intervals?
> "Accuracy is a sample statistic. Reporting it without uncertainty
> overstates precision. Bootstrap percentile intervals are non-parametric,
> cheap, and directly answer 'how much would this number change if I had
> a different test set?'"

### Q: How did you handle the small dataset?
> "Stratified split to keep class balance, fixed seed for reproducibility,
> bootstrap CIs to quantify uncertainty, and statistical significance
> testing to avoid claiming improvements that don't survive the noise."

### Q: Why not deep end-to-end training?
> "Two parts to that. (1) For 1,400 samples, end-to-end training a
> transformer would overfit catastrophically. (2) The linear probe
> protocol is what self-supervised learning papers actually use to
> measure representation quality. So I'm not avoiding deep learning;
> I'm using it in the way the field has settled on."

### Q: How does this run on a Windows laptop?
> "By design. The only step that needs a GPU is one-time embedding
> extraction in a Colab notebook. That produces a ~10 MB `.npz` file.
> Everything after that — training, evaluation, the Streamlit demo —
> runs on sklearn on the laptop CPU and takes seconds, not minutes."

### Q: Why ONNX instead of Core ML directly?
> "coremltools officially supports macOS and Linux, not Windows. ONNX
> is the cross-platform interchange format Apple's own tooling reads,
> so it's the honest path on my hardware. The Core ML conversion is
> a one-line call on any Mac — documented in `docs/coreml_path.md`."

## Numbers to memorise

(Fill these in after you run `train_all.py` — the exact numbers depend
on the random seed and your hardware.)

- **Best MFCC accuracy:** ____
- **Best wav2vec2 probe accuracy:** ____
- **Best YAMNet probe accuracy:** ____
- **Per-clip MFCC extraction time:** ~10 ms (laptop CPU)
- **Per-clip wav2vec2 extraction time:** ~50 ms (Colab T4 GPU)
- **End-to-end training time on laptop:** ____ seconds
- **ONNX file size:** ____ KB
- **ONNX vs sklearn max prob delta:** < 1e-4

## Things that might trip you up

### "Acted emotion isn't real emotion."
> "Correct, and I call that out explicitly in `docs/methodology.md` as a
> known limitation. RAVDESS is a controlled benchmark, not a deployment
> dataset. The next step would be evaluating on a more naturalistic
> dataset like IEMOCAP or MELD."

### "Why didn't you use leave-one-actor-out cross-validation?"
> "Honest answer — it's the more rigorous protocol for testing
> speaker-independence, and I document the choice not to use it. Random
> stratification matches the deployment scenario where the model has
> seen the user. Leave-one-actor-out is in the future-work list."

### "How would you scale this to streaming audio?"
> "Right now each clip is a single forward pass. For streaming you'd
> chunk the audio into overlapping windows (say 1 second every 250 ms),
> run the classifier on each, and apply temporal smoothing — which is
> what `src/voicemood/timeseries.py` already does on the analysis side.
> The harder part is voice-activity detection so you only classify
> windows that actually contain speech."

### "Did you try fine-tuning wav2vec2?"
> "I didn't, deliberately. With 1,400 training samples, the upside is
> small and the overfitting risk is large. LoRA or adapter-based
> fine-tuning would be the responsible way to do it; I list it as
> future work."

### "What's the latency of an inference call?"
> "The MFCC + sklearn pipeline: under 20 ms on a laptop CPU, dominated
> by librosa's MFCC computation. wav2vec2 forward pass on a CPU is
> 500 ms to 2 seconds depending on hardware; on Apple Silicon's Neural
> Engine it would be a few tens of milliseconds."

## What to say if asked "what would you do next?"

1. **Speaker-independent evaluation** (leave-one-actor-out).
2. **More naturalistic data** (IEMOCAP, MELD).
3. **LoRA fine-tuning of wav2vec2** instead of just linear probing.
4. **Streaming inference** with VAD and temporal smoothing.
5. **Cross-cultural evaluation** — RAVDESS is all North American English.
6. **On-device deployment** — actually ship the Core ML model in a tiny
   iOS demo app.

Have a one-sentence answer for each. Don't list all six unless asked.

## What NOT to oversell

- Don't say "publication-ready." It's a strong portfolio project, not a
  research contribution.
- Don't say "production-ready." It's a prototype with documented limits.
- Don't claim novel methods. Everything you used is from the literature.
- Don't pretend the iOS app is built. It isn't — the conversion *path*
  is documented and the ONNX export works.
