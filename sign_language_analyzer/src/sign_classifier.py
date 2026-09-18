"""
sign_classifier.py
------------------
Loads a pre-trained scikit-learn classifier and label map, then predicts
which sign a given set of hand landmarks represents.

The model and label map must both be present.  If either file is missing
or incompatible, ModelLoadError is raised at construction time so the
problem is discovered before the user tries to use the application.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import joblib
import numpy as np

from src.exceptions import (
    InvalidGestureError,
    ModelLoadError,
    PredictionError,
)


class SignClassifier:
    """
    Predicts a sign label and confidence score from normalized hand landmarks.

    Parameters
    ----------
    model_path : str
        Path to the saved scikit-learn model file (e.g. gesture_model.pkl).
    labels_path : str
        Path to the label-map JSON file (e.g. label_map.json).
    confidence_threshold : float
        Minimum probability required to accept a prediction.
        Predictions below this value are returned as ("Unknown", confidence).
    """

    def __init__(
        self,
        model_path: str,
        labels_path: str,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.model_path = model_path
        self.labels_path = labels_path
        self.confidence_threshold = confidence_threshold

        # These are set by _load(); declared here for clarity
        self._model = None
        self._label_map: Dict[int, str] = {}

        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, landmarks: List[float]) -> Tuple[str, float]:
        """
        Predict the sign for a given set of normalized landmarks.

        Parameters
        ----------
        landmarks : list[float]
            42 normalized floats from HandDetector.detect().

        Returns
        -------
        tuple[str, float]
            (label, confidence) where label is the sign name (e.g. "Hello")
            or "Unknown" if confidence < threshold.

        Raises
        ------
        PredictionError
            If the model raises an unexpected error during inference.
        InvalidGestureError
            If the predicted class index is not in the label map.
        """
        try:
            features = np.array(landmarks, dtype=np.float32).reshape(1, -1)
            proba = self._model.predict_proba(features)[0]  # shape: (n_classes,)
        except Exception as exc:
            raise PredictionError(
                f"The classifier raised an error during prediction: {exc}"
            ) from exc

        class_index = int(np.argmax(proba))
        confidence = float(proba[class_index])

        if class_index not in self._label_map:
            raise InvalidGestureError(
                f"Predicted class index {class_index} is not in the label map. "
                "The model and label map may be out of sync."
            )

        label = self._label_map[class_index]

        if confidence < self.confidence_threshold:
            return "Unknown", confidence

        return label, confidence

    def get_labels(self) -> List[str]:
        """Return the list of sign labels in class-index order."""
        return [self._label_map[i] for i in sorted(self._label_map.keys())]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """
        Load the model file and label map from disk.

        Raises
        ------
        ModelLoadError
            If either file is missing, unreadable, or incompatible.
        """
        # ── Check model file ──────────────────────────────────────────────
        if not os.path.isfile(self.model_path):
            raise ModelLoadError(
                f"Model file not found: '{self.model_path}'\n"
                "Run 'python scripts/train_model.py' to create it."
            )

        # ── Check label map file ──────────────────────────────────────────
        if not os.path.isfile(self.labels_path):
            raise ModelLoadError(
                f"Label map file not found: '{self.labels_path}'\n"
                "Run 'python scripts/train_model.py' to create it."
            )

        # ── Load model ────────────────────────────────────────────────────
        try:
            self._model = joblib.load(self.model_path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load model from '{self.model_path}': {exc}"
            ) from exc

        # ── Load label map ────────────────────────────────────────────────
        self._label_map = _load_label_map(self.labels_path)

        # ── Compatibility check ───────────────────────────────────────────
        # The model's number of classes must match the label map length.
        try:
            n_model_classes = len(self._model.classes_)
        except AttributeError:
            # Some pipelines may not expose .classes_ directly; skip check.
            n_model_classes = len(self._label_map)

        if n_model_classes != len(self._label_map):
            raise ModelLoadError(
                f"Incompatible model and label map: model has "
                f"{n_model_classes} classes but label_map.json has "
                f"{len(self._label_map)} entries. Re-train the model."
            )


# ── Module-level helper ───────────────────────────────────────────────────────

def _load_label_map(path: str) -> Dict[int, str]:
    """
    Read label_map.json and return a dict mapping integer index → sign name.

    The JSON file stores keys as strings (JSON limitation), so they are
    converted to integers here.

    Parameters
    ----------
    path : str
        Path to label_map.json.

    Returns
    -------
    dict[int, str]
        e.g. {0: "Hello", 1: "Yes", ...}

    Raises
    ------
    ModelLoadError
        If the file is not valid JSON or has unexpected structure.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw: dict = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ModelLoadError(
            f"label_map.json is not valid JSON: {exc}"
        ) from exc
    except OSError as exc:
        raise ModelLoadError(
            f"Cannot read label_map.json: {exc}"
        ) from exc

    label_map: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            label_map[int(key)] = str(value)
        except (ValueError, TypeError) as exc:
            raise ModelLoadError(
                f"label_map.json has an invalid entry: key={key!r}, "
                f"value={value!r}. Expected integer keys and string values."
            ) from exc

    if not label_map:
        raise ModelLoadError("label_map.json is empty.")

    return label_map
