"""
hand_detector.py
----------------
Wraps MediaPipe Hands to detect one hand per frame and return normalized
landmark coordinates as a flat list of floats.

The normalization step makes the output position-independent (translated
to the wrist) and scale-independent (divided by the maximum absolute value).
This means the classifier sees the same feature vector regardless of where
the hand sits in the frame or how large/small it appears.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from src.exceptions import CameraError


# MediaPipe drawing utilities used for the skeleton overlay
_mp_drawing = mp.solutions.drawing_utils
_mp_drawing_styles = mp.solutions.drawing_styles
_mp_hands = mp.solutions.hands


class HandDetector:
    """
    Detects a hand in a BGR video frame and returns 42 normalized floats
    (21 landmarks × x and y coordinates).

    Parameters
    ----------
    min_detection_confidence : float
        MediaPipe minimum detection confidence (default 0.7).
    max_num_hands : int
        Maximum number of hands to detect. MVP uses 1.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        max_num_hands: int = 1,
    ) -> None:
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> Optional[List[float]]:
        """
        Process one BGR frame and return normalized hand landmarks.

        Parameters
        ----------
        frame : np.ndarray
            BGR image from OpenCV (as returned by cap.read()).

        Returns
        -------
        list[float] or None
            A flat list of 42 floats [x0, y0, x1, y1, …, x20, y20] where
            each coordinate is wrist-relative and scale-normalized.
            Returns None if no hand is detected.

        Raises
        ------
        CameraError
            If the frame is None or has an unexpected shape.
        """
        if frame is None:
            raise CameraError("Received a None frame from the camera.")

        # MediaPipe expects RGB; OpenCV provides BGR
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return None  # No hand detected — caller shows "No hand detected"

        # Use only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        return _normalize_landmarks(hand_landmarks)

    def draw_landmarks(
        self, frame: np.ndarray, results: object
    ) -> np.ndarray:
        """
        Draw the hand skeleton overlay on a copy of the frame.

        Parameters
        ----------
        frame : np.ndarray
            The BGR frame to annotate.
        results : object
            The raw MediaPipe Hands results object returned by process().

        Returns
        -------
        np.ndarray
            A new frame with the hand skeleton drawn on it.
        """
        annotated = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_drawing_styles.get_default_hand_landmarks_style(),
                    _mp_drawing_styles.get_default_hand_connections_style(),
                )
        return annotated

    def process_for_display(self, frame: np.ndarray):
        """
        Run MediaPipe and return both the landmark list and the annotated frame.

        This combines detect() and draw_landmarks() in a single call so that
        app.py only needs to call one method per frame.

        Returns
        -------
        tuple[list[float] | None, np.ndarray]
            (landmarks, annotated_frame)
        """
        if frame is None:
            raise CameraError("Received a None frame from the camera.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        annotated = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                _mp_drawing.draw_landmarks(
                    annotated,
                    hand_landmarks,
                    _mp_hands.HAND_CONNECTIONS,
                    _mp_drawing_styles.get_default_hand_landmarks_style(),
                    _mp_drawing_styles.get_default_hand_connections_style(),
                )
            landmarks = _normalize_landmarks(results.multi_hand_landmarks[0])
        else:
            landmarks = None

        return landmarks, annotated

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()


# ── Module-level helpers ──────────────────────────────────────────────────────

def _normalize_landmarks(hand_landmarks) -> List[float]:
    """
    Convert raw MediaPipe hand landmarks to a normalized flat list.

    Steps
    -----
    1. Extract (x, y) for each of the 21 keypoints.
    2. Subtract the wrist position (landmark index 0) so the hand is
       position-independent.
    3. Divide by the maximum absolute value so the hand is scale-independent.
    4. Flatten to a 1-D list of 42 floats.

    Parameters
    ----------
    hand_landmarks : mediapipe NormalizedLandmarkList
        The raw landmark object from MediaPipe.

    Returns
    -------
    list[float]
        42 normalized floats.
    """
    coords = np.array(
        [[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32
    )  # shape: (21, 2)

    # Translate: make wrist the origin
    coords -= coords[0]

    # Scale: divide by the max absolute value to fit in [-1, 1]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val

    return coords.flatten().tolist()
