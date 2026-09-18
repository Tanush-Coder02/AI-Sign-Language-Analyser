"""
exceptions.py
-------------
All custom exceptions for the AI Sign Language Analyzer.

Every exception inherits from AppError so callers can catch the entire
family with a single `except AppError` clause if needed.
"""


class AppError(Exception):
    """Base class for all application-specific exceptions."""


class CameraError(AppError):
    """
    Raised when the webcam cannot be opened or a frame cannot be read.

    Example situations:
    - cv2.VideoCapture(0) returns False for isOpened()
    - A frame read mid-session returns None
    - The camera is disconnected during an active session
    """


class ModelLoadError(AppError):
    """
    Raised when the trained model or label map cannot be loaded.

    Example situations:
    - The .pkl file does not exist
    - The JSON label map is malformed
    - The number of classes in the model does not match the label map
    """


class PredictionError(AppError):
    """
    Raised when the classifier raises an unexpected error during inference.

    This is distinct from a low-confidence prediction, which is handled
    by the SignClassifier returning ("Unknown", confidence) instead.
    """


class InvalidGestureError(AppError):
    """
    Raised when a predicted class index is not present in the label map.

    This should not happen if the model and label map were saved together,
    but it is checked at runtime as a safety guard.
    """


class SpeechError(AppError):
    """
    Raised when the text-to-speech engine fails to initialize or speak.

    Example situations:
    - pyttsx3 cannot find a TTS driver on the current OS
    - The engine raises an error while running speech
    """


class ConfigurationError(AppError):
    """
    Raised when a configuration value is missing, has the wrong type,
    or falls outside the allowed range.
    """
