"""Export trained sklearn pipelines to ONNX format.

ONNX is the cross-platform interchange format for ML models. From ONNX we
can convert to:
    - Core ML (Apple devices) via coremltools.converters.onnx
    - TensorFlow Lite (Android) via onnx-tf
    - ONNX Runtime (cross-platform inference) directly

Note on choice: sklearn -> ONNX via skl2onnx works on Windows / macOS / Linux.
Direct sklearn -> Core ML conversion is officially supported on macOS only.
ONNX as the intermediate format makes the project Windows-friendly while
preserving the Apple deployment story.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_pipeline_to_onnx(
    pipeline_path: Path,
    onnx_path: Path,
    *,
    n_features: int,
    model_name: str = "voicemood",
) -> Path:
    """Convert a saved sklearn Pipeline to an ONNX file.

    Args:
        pipeline_path: path to a joblib-dumped sklearn Pipeline.
        onnx_path: where to write the resulting .onnx file.
        n_features: input feature dimension (needed for ONNX type spec).
        model_name: graph name embedded in the ONNX file.

    Returns:
        Path to the written ONNX file.
    """
    pipeline = joblib.load(pipeline_path)

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        pipeline,
        name=model_name,
        initial_types=initial_type,
        target_opset=15,
        options={id(pipeline): {"zipmap": False}},  # plain tensor outputs
    )

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with onnx_path.open("wb") as f:
        f.write(onnx_model.SerializeToString())

    return onnx_path


def verify_onnx_matches_sklearn(
    sklearn_pipeline_path: Path,
    onnx_path: Path,
    X_sample,
    atol: float = 1e-4,
) -> dict:
    """Sanity-check: run the same input through sklearn and ONNX Runtime.

    Returns a dict with the maximum absolute difference between the two
    predicted probability vectors. atol is the tolerance you'd typically
    use to call them "equivalent".
    """
    import numpy as np
    import onnxruntime as ort

    pipeline = joblib.load(sklearn_pipeline_path)
    sklearn_proba = pipeline.predict_proba(X_sample.astype("float32"))

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: X_sample.astype("float32")})

    # skl2onnx returns [labels, probabilities] in that order by default.
    onnx_proba = outputs[1] if len(outputs) > 1 else outputs[0]

    max_diff = float(np.max(np.abs(sklearn_proba - onnx_proba)))
    return {"max_abs_diff": max_diff, "passes_tolerance": max_diff <= atol}
