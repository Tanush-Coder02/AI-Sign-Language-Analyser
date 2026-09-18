"""
test_stability_tracker.py
-------------------------
Unit tests for StabilityTracker.

No camera or AI dependencies required.
"""

import pytest

from src.stability_tracker import StabilityTracker


class TestStabilityAccepted:
    """Tests for when update() should return True."""

    def test_returns_true_after_required_frames(self):
        tracker = StabilityTracker(required_frames=3)
        assert tracker.update("Hello") is False
        assert tracker.update("Hello") is False
        assert tracker.update("Hello") is True  # 3rd frame → True

    def test_returns_true_on_every_subsequent_frame(self):
        """Once stable, keep returning True if the same sign continues."""
        tracker = StabilityTracker(required_frames=2)
        tracker.update("Yes")
        assert tracker.update("Yes") is True
        assert tracker.update("Yes") is True  # buffer is full, still same sign

    def test_required_frames_of_one(self):
        tracker = StabilityTracker(required_frames=1)
        assert tracker.update("Stop") is True


class TestStabilityReset:
    """Tests for when a different sign resets the counter."""

    def test_different_sign_resets_buffer(self):
        tracker = StabilityTracker(required_frames=3)
        tracker.update("Hello")
        tracker.update("Hello")
        # A different sign — the buffer should clear
        result = tracker.update("Yes")
        assert result is False
        assert tracker.current_count == 1

    def test_alternating_signs_never_stable(self):
        tracker = StabilityTracker(required_frames=3)
        for _ in range(10):
            assert tracker.update("Hello") is False
            assert tracker.update("Yes") is False

    def test_almost_stable_then_different(self):
        tracker = StabilityTracker(required_frames=5)
        for _ in range(4):
            tracker.update("Water")
        # One different sign interrupts; buffer clears, "Food" count = 1
        tracker.update("Food")
        # Must reach 5 total "Food" frames; we already have 1, need 4 more
        for i in range(3):
            result = tracker.update("Food")
            assert result is False  # frames 2, 3, 4
        assert tracker.update("Food") is True  # frame 5


class TestReset:
    """Tests for StabilityTracker.reset()."""

    def test_manual_reset_clears_buffer(self):
        tracker = StabilityTracker(required_frames=3)
        tracker.update("Hello")
        tracker.update("Hello")
        tracker.reset()
        assert tracker.current_count == 0
        assert tracker.update("Hello") is False  # needs N frames again

    def test_reset_then_same_sign_requires_full_count(self):
        tracker = StabilityTracker(required_frames=2)
        tracker.update("No")
        tracker.update("No")  # True here
        tracker.reset()
        assert tracker.update("No") is False
        assert tracker.update("No") is True


class TestCurrentCount:
    """Tests for the current_count property."""

    def test_count_increments(self):
        tracker = StabilityTracker(required_frames=5)
        assert tracker.current_count == 0
        tracker.update("Help")
        assert tracker.current_count == 1
        tracker.update("Help")
        assert tracker.current_count == 2

    def test_count_resets_on_sign_change(self):
        tracker = StabilityTracker(required_frames=5)
        tracker.update("Hello")
        tracker.update("Hello")
        tracker.update("Yes")   # different sign
        assert tracker.current_count == 1


class TestConfiguration:
    """Tests for StabilityTracker constructor validation."""

    def test_zero_required_frames_raises(self):
        with pytest.raises(ValueError):
            StabilityTracker(required_frames=0)

    def test_negative_required_frames_raises(self):
        with pytest.raises(ValueError):
            StabilityTracker(required_frames=-1)

    def test_non_integer_required_frames_raises(self):
        with pytest.raises(ValueError):
            StabilityTracker(required_frames=2.5)
