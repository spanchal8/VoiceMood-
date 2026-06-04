"""sklearn classifier factory.

We expose one `build_classifier` function that returns a fully configured
sklearn Pipeline (StandardScaler + classifier). Wrapping with a Pipeline
means the entire model — including the scaler — is one ONNX-exportable
object.

Three model families:
    - logistic_regression : linear, calibrated, interpretable baseline
    - random_forest       : non-linear, captures feature interactions
    - svm_rbf             : kernel method, often strong on small datasets
"""

from __future__ import annotations

from typing import Literal

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import RANDOM_SEED

ModelName = Literal["logistic_regression", "random_forest", "svm_rbf"]


def build_classifier(name: ModelName) -> Pipeline:
    """Return a sklearn Pipeline ready to .fit(X, y).

    All classifiers expose .predict_proba thanks to either built-in
    probability outputs or CalibratedClassifierCV wrapping.
    """
    if name == "logistic_regression":
        # Note: sklearn >= 1.5 defaults to multinomial for multi-class problems
        # with the lbfgs solver, so we don't need to set multi_class explicitly
        # (and the argument was deprecated).
        clf = LogisticRegression(
            max_iter=2_000,
            solver="lbfgs",
            C=1.0,
            random_state=RANDOM_SEED,
        )
    elif name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        )
    elif name == "svm_rbf":
        # SVM with probability=True does internal Platt scaling, but it's
        # slow. CalibratedClassifierCV around LinearSVC would be faster,
        # but for RAVDESS (~1440 samples) probability=True is fine.
        clf = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED,
        )
    else:
        raise ValueError(f"Unknown model name: {name!r}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def all_model_names() -> list[ModelName]:
    """Return the canonical ordering of model names for evaluation tables."""
    return ["logistic_regression", "random_forest", "svm_rbf"]
