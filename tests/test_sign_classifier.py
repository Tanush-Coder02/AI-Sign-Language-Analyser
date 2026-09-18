"""
test_sign_classifier.py
-----------------------
Unit tests for SignClassifier.

The real model and label map files are not required — all file loading
and sklearn calls are mocked using pytest-mock.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from src.exceptions import (
    InvalidGestureError,
    ModelLoadError,
    PredictionError,
)
from src.sign_classifier import SignClassifier, _load_label_map


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_label_map(tmp_path):
    """Write a minimal label_map.json to a temp directory and return its path."""
    label_map = {"0": "Hello", "1": "Yes", "2": "Stop"}
    path = tmp_path / "label_map.json"
    path.write_text(json.dumps(label_map), encoding="utf-8")
    return str(path)


@pytest.fixture
def mock_model(mocker):
    """Return a mock scikit-learn model with predict_proba."""
    m = mocker.MagicMock()
    m.classes_ = [0, 1, 2]
    # Default: predict "Hello" (index 0) with 90% confidence
    m.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
    return m


@pytest.fixture
def classifier(tmp_path, tmp_label_map, mock_model, mocker):
    """Build a SignClassifier with mocked file loading."""
    model_path = str(tmp_path / "gesture_model.pkl")
    # Create an empty file so os.path.isfile() passes
    open(model_path, "w").close()

    mocker.patch("src.sign_classifier.joblib.load", return_value=mock_model)

    clf = SignClassifier(
        model_path=model_path,
        labels_path=tmp_label_map,
        confidence_threshold=0.75,
    )
    return clf, mock_model


# ── Tests: predict() ──────────────────────────────────────────────────────────

class TestPredict:

    def test_high_confidence_returns_label(self, classifier):
        clf, mock_model = classifier
        mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
        label, conf = clf.predict([0.0] * 42)
        assert label == "Hello"
        assert abs(conf - 0.9) < 1e-6

    def test_confidence_below_threshold_returns_unknown(self, classifier):
        clf, mock_model = classifier
        # 60% confidence — below the 75% threshold
        mock_model.predict_proba.return_value = np.array([[0.6, 0.2, 0.2]])
        label, conf = clf.predict([0.0] * 42)
        assert label == "Unknown"
        assert abs(conf - 0.6) < 1e-6

    def test_confidence_equal_to_threshold_accepted(self, classifier):
        """Confidence exactly equal to threshold should be accepted (≥ comparison)."""
        clf, mock_model = classifier
        mock_model.predict_proba.return_value = np.array([[0.75, 0.15, 0.10]])
        label, conf = clf.predict([0.0] * 42)
        assert label == "Hello"

    def test_second_class_predicted(self, classifier):
        clf, mock_model = classifier
        mock_model.predict_proba.return_value = np.array([[0.05, 0.9, 0.05]])
        label, conf = clf.predict([0.0] * 42)
        assert label == "Yes"

    def test_prediction_error_raised_on_model_exception(self, classifier):
        clf, mock_model = classifier
        mock_model.predict_proba.side_effect = RuntimeError("model exploded")
        with pytest.raises(PredictionError, match="classifier raised an error"):
            clf.predict([0.0] * 42)


# ── Tests: invalid gesture ────────────────────────────────────────────────────

class TestInvalidGesture:

    def test_class_index_not_in_label_map_raises(self, tmp_path, mocker):
        """If the model predicts a class not in the label map, raise InvalidGestureError."""
        label_map = {"0": "Hello"}  # only 1 entry
        lm_path = str(tmp_path / "label_map.json")
        with open(lm_path, "w") as f:
            json.dump(label_map, f)

        model_path = str(tmp_path / "model.pkl")
        open(model_path, "w").close()

        mock_m = mocker.MagicMock()
        mock_m.classes_ = [0]  # matches label_map length (1)
        # But predict index 5, which is NOT in label_map
        mock_m.predict_proba.return_value = np.array([[1.0]])

        mocker.patch("src.sign_classifier.joblib.load", return_value=mock_m)

        clf = SignClassifier(model_path, lm_path, confidence_threshold=0.5)
        # Manually break the label_map to simulate the impossible scenario
        clf._label_map = {}  # empty → any predicted index is invalid
        with pytest.raises(InvalidGestureError):
            clf.predict([0.0] * 42)


# ── Tests: model loading ──────────────────────────────────────────────────────

class TestModelLoading:

    def test_missing_model_file_raises(self, tmp_path, tmp_label_map):
        with pytest.raises(ModelLoadError, match="Model file not found"):
            SignClassifier(
                model_path=str(tmp_path / "nonexistent.pkl"),
                labels_path=tmp_label_map,
            )

    def test_missing_label_map_raises(self, tmp_path, mocker):
        model_path = str(tmp_path / "model.pkl")
        open(model_path, "w").close()
        with pytest.raises(ModelLoadError, match="Label map file not found"):
            SignClassifier(
                model_path=model_path,
                labels_path=str(tmp_path / "nonexistent.json"),
            )

    def test_mismatched_class_count_raises(self, tmp_path, tmp_label_map, mocker):
        """Model has 5 classes but label map has 3 — should raise ModelLoadError."""
        model_path = str(tmp_path / "model.pkl")
        open(model_path, "w").close()

        bad_model = mocker.MagicMock()
        bad_model.classes_ = list(range(5))  # 5 classes
        mocker.patch("src.sign_classifier.joblib.load", return_value=bad_model)

        with pytest.raises(ModelLoadError, match="Incompatible"):
            SignClassifier(
                model_path=model_path,
                labels_path=tmp_label_map,  # has 3 classes
            )

    def test_corrupt_label_map_raises(self, tmp_path, mocker):
        model_path = str(tmp_path / "model.pkl")
        lm_path = str(tmp_path / "bad_labels.json")
        open(model_path, "w").close()
        with open(lm_path, "w") as f:
            f.write("{not valid json}")
        with pytest.raises(ModelLoadError):
            SignClassifier(model_path=model_path, labels_path=lm_path)


# ── Tests: _load_label_map helper ────────────────────────────────────────────

class TestLoadLabelMap:

    def test_valid_label_map(self, tmp_path):
        path = tmp_path / "lm.json"
        path.write_text(json.dumps({"0": "Hello", "1": "Yes"}))
        result = _load_label_map(str(path))
        assert result == {0: "Hello", 1: "Yes"}

    def test_empty_label_map_raises(self, tmp_path):
        path = tmp_path / "lm.json"
        path.write_text("{}")
        with pytest.raises(ModelLoadError, match="empty"):
            _load_label_map(str(path))

    def test_invalid_json_raises(self, tmp_path):
        path = tmp_path / "lm.json"
        path.write_text("not json")
        with pytest.raises(ModelLoadError):
            _load_label_map(str(path))


# ── Tests: get_labels() ───────────────────────────────────────────────────────

class TestGetLabels:

    def test_get_labels_returns_ordered_list(self, classifier):
        clf, _ = classifier
        labels = clf.get_labels()
        assert labels == ["Hello", "Yes", "Stop"]
