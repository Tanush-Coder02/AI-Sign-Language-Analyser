"""
config.py
---------
Loads and validates application configuration from a JSON file.

Configuration is stored in config.json at the project root.
If the file is missing, sensible defaults are used and a warning is printed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import List

from src.exceptions import ConfigurationError


# ── Default values used when config.json is absent ───────────────────────────
_DEFAULTS = {
    "confidence_threshold": 0.75,
    "stability_frames": 10,
    "cooldown_seconds": 2.0,
    "max_num_hands": 1,
    "vocabulary": [
        "Hello", "Yes", "No", "Help",
        "Thank You", "Water", "Food", "Stop",
    ],
}


@dataclass
class AppConfig:
    """
    Holds all runtime configuration values.

    Fields
    ------
    confidence_threshold : float
        Minimum probability (0 < value ≤ 1) for a prediction to be accepted.
    stability_frames : int
        Number of consecutive identical predictions required before a sign
        is considered stable and eligible to be added to the sentence.
    cooldown_seconds : float
        Minimum time in seconds that must pass before the same sign can be
        added to the sentence again.
    max_num_hands : int
        Maximum number of hands to detect (1 or 2). MVP uses 1.
    vocabulary : list[str]
        The ordered list of sign labels this application supports.
    """

    confidence_threshold: float = 0.75
    stability_frames: int = 10
    cooldown_seconds: float = 2.0
    max_num_hands: int = 1
    vocabulary: List[str] = field(default_factory=lambda: list(_DEFAULTS["vocabulary"]))

    # ── Class-method constructor ──────────────────────────────────────────────

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        """
        Load configuration from a JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON configuration file.

        Returns
        -------
        AppConfig
            A validated configuration object.

        Raises
        ------
        ConfigurationError
            If any field has an invalid type or value.
        """
        if not os.path.isfile(path):
            # Missing file is acceptable — use defaults, show a warning.
            print(
                f"[WARNING] config.json not found at '{path}'. "
                "Using default configuration."
            )
            return cls()

        with open(path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ConfigurationError(
                    f"config.json is not valid JSON: {exc}"
                ) from exc

        # Fill missing keys with defaults before validation
        merged = {**_DEFAULTS, **data}
        _validate(merged)

        return cls(
            confidence_threshold=float(merged["confidence_threshold"]),
            stability_frames=int(merged["stability_frames"]),
            cooldown_seconds=float(merged["cooldown_seconds"]),
            max_num_hands=int(merged["max_num_hands"]),
            vocabulary=list(merged["vocabulary"]),
        )


# ── Internal validation helper ────────────────────────────────────────────────

def _validate(data: dict) -> None:
    """
    Check that every configuration value is within the allowed range.

    Raises ConfigurationError with a descriptive message on the first
    violation found.
    """
    ct = data.get("confidence_threshold")
    if not isinstance(ct, (int, float)) or not (0.0 < float(ct) <= 1.0):
        raise ConfigurationError(
            f"confidence_threshold must be a float in (0, 1]. Got: {ct!r}"
        )

    sf = data.get("stability_frames")
    if not isinstance(sf, int) or sf < 1:
        raise ConfigurationError(
            f"stability_frames must be a positive integer (≥ 1). Got: {sf!r}"
        )

    cs = data.get("cooldown_seconds")
    if not isinstance(cs, (int, float)) or float(cs) < 0.0:
        raise ConfigurationError(
            f"cooldown_seconds must be a non-negative float. Got: {cs!r}"
        )

    mnh = data.get("max_num_hands")
    if mnh not in (1, 2):
        raise ConfigurationError(
            f"max_num_hands must be 1 or 2. Got: {mnh!r}"
        )

    vocab = data.get("vocabulary")
    if not isinstance(vocab, list) or len(vocab) == 0:
        raise ConfigurationError(
            "vocabulary must be a non-empty list of strings."
        )
    for item in vocab:
        if not isinstance(item, str) or not item.strip():
            raise ConfigurationError(
                f"Every vocabulary entry must be a non-empty string. Got: {item!r}"
            )
