# ONNX → Core ML Conversion Path

VoiceMood exports the trained sklearn pipeline to ONNX (cross-platform).
This document describes the documented path from ONNX to Apple Core ML for
on-device deployment on iOS / macOS / iPadOS / watchOS.

## Why ONNX as the intermediate

`coremltools` is officially supported on macOS and Linux only. On a
Windows development machine, the cleanest route is:

```
sklearn Pipeline  --[ skl2onnx ]-->  ONNX  --[ coremltools ]-->  .mlmodel
       (Windows)                  (cross-plat)             (macOS step)
```

ONNX is a stable, well-supported interchange format used by Apple itself.
Apple's own conversion tooling reads ONNX natively.

## Step-by-step

### 1. Produce the ONNX file (Windows / any platform)

```bash
python scripts/export_onnx.py
```

This writes `results/voicemood.onnx` (a few hundred KB) and runs a
sklearn-vs-ONNX consistency check on a held-out batch to verify the
conversion preserves probabilities to within 1e-4.

### 2. Convert ONNX to Core ML (macOS recommended)

On any Mac, with Python 3.10+:

```bash
pip install coremltools onnx
```

```python
import coremltools as ct

mlmodel = ct.converters.onnx.convert(
    model="voicemood.onnx",
    minimum_ios_deployment_target="13",
)

mlmodel.author = "Shubh Panchal"
mlmodel.short_description = (
    "Speech emotion recognition from 170-dim acoustic feature vector. "
    "8 classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised."
)
mlmodel.input_description["input"] = "170-dim acoustic feature vector (float32)."
mlmodel.output_description["probabilities"] = "Per-class probabilities (float32)."

mlmodel.save("VoiceMood.mlmodel")
```

The resulting `VoiceMood.mlmodel` can be dropped into an Xcode project
and used directly.

### 3. Compute MFCC features on iOS

The Core ML model expects a 170-dim feature vector. Two options on the
Swift side:

**Option A: AVFoundation + custom DSP**
Compute MFCCs in Swift using `vDSP` / `Accelerate.framework`. There are
several open-source Swift MFCC implementations.

**Option B: Embed librosa-equivalent in a second model**
Train a Keras / PyTorch front-end that mimics librosa's MFCC and convert
that to Core ML too, so the chain is `audio → CoreML(features) →
CoreML(classifier)`.

Option A is simpler and recommended; Option B is heavier but completely
removes any signal-processing reimplementation risk.

### 4. Quantisation for size and battery

```python
import coremltools.optimize.coreml as cto

# 8-bit linear quantisation typically gives < 1% accuracy delta on
# logistic regression / random forest pipelines and roughly 4x file
# size reduction.
config = cto.OptimizationConfig(
    global_config=cto.OpLinearQuantizerConfig(mode="linear_symmetric", weight_threshold=512)
)
quantised = cto.linear_quantize_weights(mlmodel, config=config)
quantised.save("VoiceMood-int8.mlmodel")
```

## Notes on deep-model conversion

The wav2vec2 and YAMNet pipelines in this project live upstream in the
Colab notebook and are not part of the on-device deployment path here.
If on-device deep-model inference were desired:

- **wav2vec2 → Core ML**: HuggingFace's `exporters` package supports this
  via `optimum`. Expect a ~95 MB Core ML file for the base model. Apple's
  Neural Engine handles transformer encoders reasonably well as of A14
  and later.
- **YAMNet → Core ML**: TensorFlow Lite is the lighter route; Core ML
  conversion from a SavedModel via `tfcoreml` or `coremltools` >= 6 is
  possible but flakier than ONNX.

The current MFCC + sklearn deployment fits in well under 1 MB and runs in
sub-millisecond time on any iPhone since the iPhone 6s. It's the right
trade-off for this project's scale.

## Reference

- Apple coremltools documentation: https://apple.github.io/coremltools
- ONNX format specification: https://onnx.ai
- skl2onnx documentation: https://onnx.ai/sklearn-onnx
