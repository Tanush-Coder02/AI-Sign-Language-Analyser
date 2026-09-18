"""
test_config.py
--------------
Unit tests for AppConfig and the _validate helper.

No camera, model, or TTS dependencies.
"""

import json
import os

import pytest

from src.config import AppConfig, _validate
from src.exceptions import ConfigurationError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def valid_config_data():
    """Return a dict that should pass validation without errors."""
    return {
        "confidence_threshold": 0.75,
        "stability_frames": 10,
        "cooldown_seconds": 2.0,
        "max_num_hands": 1,
        "vocabulary": ["Hello", "Yes", "No"],
    }


@pytest.fixture
def valid_config_file(tmp_path, valid_config_data):
    """Write valid config data to a temp JSON file and return the path."""
    path = tmp_path / "config.json"
    path.write_text(json.dumps(valid_config_data), encoding="utf-8")
    return str(path)


# ── Tests: AppConfig.load() with a valid file ─────────────────────────────────

class TestLoadValidConfig:

    def test_loads_all_fields_correctly(self, valid_config_file):
        cfg = AppConfig.load(valid_config_file)
        assert cfg.confidence_threshold == 0.75
        assert cfg.stability_frames == 10
        assert cfg.cooldown_seconds == 2.0
        assert cfg.max_num_hands == 1
        assert cfg.vocabulary == ["Hello", "Yes", "No"]

    def test_missing_file_returns_defaults(self, tmp_path):
        """A missing config.json should not raise — defaults are applied."""
        cfg = AppConfig.load(str(tmp_path / "nonexistent.json"))
        assert cfg.confidence_threshold == 0.75
        assert cfg.stability_frames == 10
        assert cfg.vocabulary  # not empty

    def test_corrupt_json_raises_configuration_error(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("{not valid json}")
        with pytest.raises(ConfigurationError, match="not valid JSON"):
            AppConfig.load(str(path))


# ── Tests: confidence_threshold validation ────────────────────────────────────

class TestConfidenceThresholdValidation:

    def test_zero_confidence_threshold_raises(self, valid_config_data):
        valid_config_data["confidence_threshold"] = 0.0
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            _validate(valid_config_data)

    def test_negative_confidence_threshold_raises(self, valid_config_data):
        valid_config_data["confidence_threshold"] = -0.1
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            _validate(valid_config_data)

    def test_confidence_threshold_above_one_raises(self, valid_config_data):
        valid_config_data["confidence_threshold"] = 1.1
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            _validate(valid_config_data)

    def test_confidence_threshold_of_one_is_valid(self, valid_config_data):
        valid_config_data["confidence_threshold"] = 1.0
        _validate(valid_config_data)  # should not raise

    def test_confidence_threshold_string_raises(self, valid_config_data):
        valid_config_data["confidence_threshold"] = "high"
        with pytest.raises(ConfigurationError, match="confidence_threshold"):
            _validate(valid_config_data)


# ── Tests: stability_frames validation ───────────────────────────────────────

class TestStabilityFramesValidation:

    def test_zero_stability_frames_raises(self, valid_config_data):
        valid_config_data["stability_frames"] = 0
        with pytest.raises(ConfigurationError, match="stability_frames"):
            _validate(valid_config_data)

    def test_negative_stability_frames_raises(self, valid_config_data):
        valid_config_data["stability_frames"] = -5
        with pytest.raises(ConfigurationError, match="stability_frames"):
            _validate(valid_config_data)

    def test_float_stability_frames_raises(self, valid_config_data):
        valid_config_data["stability_frames"] = 5.5
        with pytest.raises(ConfigurationError, match="stability_frames"):
            _validate(valid_config_data)


# ── Tests: cooldown_seconds validation ───────────────────────────────────────

class TestCooldownValidation:

    def test_negative_cooldown_raises(self, valid_config_data):
        valid_config_data["cooldown_seconds"] = -1.0
        with pytest.raises(ConfigurationError, match="cooldown_seconds"):
            _validate(valid_config_data)

    def test_zero_cooldown_is_valid(self, valid_config_data):
        valid_config_data["cooldown_seconds"] = 0.0
        _validate(valid_config_data)  # should not raise


# ── Tests: max_num_hands validation ──────────────────────────────────────────

class TestMaxNumHandsValidation:

    def test_zero_max_hands_raises(self, valid_config_data):
        valid_config_data["max_num_hands"] = 0
        with pytest.raises(ConfigurationError, match="max_num_hands"):
            _validate(valid_config_data)

    def test_three_max_hands_raises(self, valid_config_data):
        valid_config_data["max_num_hands"] = 3
        with pytest.raises(ConfigurationError, match="max_num_hands"):
            _validate(valid_config_data)

    def test_two_max_hands_is_valid(self, valid_config_data):
        valid_config_data["max_num_hands"] = 2
        _validate(valid_config_data)  # should not raise


# ── Tests: vocabulary validation ─────────────────────────────────────────────

class TestVocabularyValidation:

    def test_empty_vocabulary_raises(self, valid_config_data):
        valid_config_data["vocabulary"] = []
        with pytest.raises(ConfigurationError, match="vocabulary"):
            _validate(valid_config_data)

    def test_vocabulary_with_empty_string_raises(self, valid_config_data):
        valid_config_data["vocabulary"] = ["Hello", ""]
        with pytest.raises(ConfigurationError, match="vocabulary"):
            _validate(valid_config_data)

    def test_vocabulary_with_whitespace_string_raises(self, valid_config_data):
        valid_config_data["vocabulary"] = ["Hello", "   "]
        with pytest.raises(ConfigurationError, match="vocabulary"):
            _validate(valid_config_data)

    def test_vocabulary_not_a_list_raises(self, valid_config_data):
        valid_config_data["vocabulary"] = "Hello Yes No"
        with pytest.raises(ConfigurationError, match="vocabulary"):
            _validate(valid_config_data)
