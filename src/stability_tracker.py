"""
stability_tracker.py
--------------------
Implements a simple frame-buffer that returns True only after the same
sign label has been predicted for N consecutive frames.

This prevents jitter: a single uncertain or mis-predicted frame will not
cause a word to be added to the sentence.  The user must hold a pose
steadily for `required_frames` frames in a row.
"""

from __future__ import annotations

from collections import deque


class StabilityTracker:
    """
    Tracks whether the most recent N predictions are all the same sign.

    Parameters
    ----------
    required_frames : int
        Number of identical consecutive predictions needed before update()
        returns True.  Must be a positive integer.

    Usage
    -----
    tracker = StabilityTracker(required_frames=10)

    # Inside the webcam loop:
    is_stable = tracker.update(current_label)
    if is_stable:
        sentence_builder.add_word(current_label)
        tracker.reset()   # optional: prevent adding the word every frame
    """

    def __init__(self, required_frames: int) -> None:
        if not isinstance(required_frames, int) or required_frames < 1:
            raise ValueError(
                f"required_frames must be a positive integer. Got: {required_frames!r}"
            )
        self.required_frames = required_frames
        # The deque has a fixed maximum length; old values fall off automatically
        self._buffer: deque = deque(maxlen=required_frames)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, sign: str) -> bool:
        """
        Add the latest prediction to the buffer and check for stability.

        If any previous entry differs from `sign`, the buffer is cleared
        first so the count restarts from 1.

        Parameters
        ----------
        sign : str
            The predicted sign label for the current frame.

        Returns
        -------
        bool
            True if the buffer is full and every entry equals `sign`.
        """
        # If the sign changed, start a fresh count
        if self._buffer and self._buffer[-1] != sign:
            self._buffer.clear()

        self._buffer.append(sign)

        # True only when all N slots are filled with the same sign
        return (
            len(self._buffer) == self.required_frames
            and len(set(self._buffer)) == 1
        )

    def reset(self) -> None:
        """
        Clear the buffer.

        Call this after a sign has been accepted so that the same sign
        cannot be accepted again on the very next frame.
        """
        self._buffer.clear()

    @property
    def current_count(self) -> int:
        """Number of identical frames accumulated so far."""
        return len(self._buffer)
