"""
test_sentence_builder.py
------------------------
Unit tests for SentenceBuilder.

These tests do not require a camera, model, or TTS engine.
"""

import time

import pytest

from src.sentence_builder import SentenceBuilder


class TestAddWord:
    """Tests for SentenceBuilder.add_word()."""

    def test_add_valid_word(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        result = builder.add_word("Hello")
        assert result is True
        assert builder.get_sentence() == "Hello"

    def test_add_multiple_different_words(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        builder.add_word("Hello")
        builder.add_word("Water")
        builder.add_word("Stop")
        assert builder.get_sentence() == "Hello Water Stop"

    def test_empty_string_rejected(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        result = builder.add_word("")
        assert result is False
        assert builder.is_empty()

    def test_whitespace_only_rejected(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        result = builder.add_word("   ")
        assert result is False
        assert builder.is_empty()

    def test_unknown_label_rejected(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        result = builder.add_word("Unknown")
        assert result is False
        assert builder.is_empty()

    def test_unknown_label_case_insensitive(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        result = builder.add_word("UNKNOWN")
        assert result is False


class TestDuplicatePrevention:
    """Tests for cooldown-based duplicate prevention."""

    def test_same_word_within_cooldown_rejected(self):
        builder = SentenceBuilder(cooldown_seconds=5.0)
        builder.add_word("Hello")
        result = builder.add_word("Hello")
        assert result is False
        assert builder.get_sentence() == "Hello"

    def test_same_word_after_cooldown_accepted(self):
        builder = SentenceBuilder(cooldown_seconds=0.05)
        builder.add_word("Hello")
        time.sleep(0.1)
        result = builder.add_word("Hello")
        assert result is True
        assert builder.get_sentence() == "Hello Hello"

    def test_different_word_not_blocked_by_cooldown(self):
        builder = SentenceBuilder(cooldown_seconds=5.0)
        builder.add_word("Hello")
        result = builder.add_word("Water")
        assert result is True
        assert builder.get_sentence() == "Hello Water"

    def test_zero_cooldown_allows_immediate_repeat(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        builder.add_word("Yes")
        result = builder.add_word("Yes")
        assert result is True


class TestRemoveLast:
    """Tests for SentenceBuilder.remove_last()."""

    def test_remove_last_removes_word(self):
        builder = SentenceBuilder()
        builder.add_word("Hello")
        builder.add_word("Water")
        removed = builder.remove_last()
        assert removed is True
        assert builder.get_sentence() == "Hello"

    def test_remove_last_on_empty_returns_false(self):
        builder = SentenceBuilder()
        result = builder.remove_last()
        assert result is False
        assert builder.is_empty()

    def test_remove_last_allows_same_word_again(self):
        """After remove_last, the last-sign cooldown is reset."""
        builder = SentenceBuilder(cooldown_seconds=5.0)
        builder.add_word("Hello")
        builder.remove_last()
        # Should be accepted because tracking was reset
        result = builder.add_word("Hello")
        assert result is True


class TestClear:
    """Tests for SentenceBuilder.clear()."""

    def test_clear_empties_sentence(self):
        builder = SentenceBuilder()
        builder.add_word("Hello")
        builder.add_word("Yes")
        builder.clear()
        assert builder.is_empty()
        assert builder.get_sentence() == ""

    def test_clear_on_empty_sentence_no_error(self):
        builder = SentenceBuilder()
        builder.clear()  # Should not raise
        assert builder.is_empty()


class TestGetSentence:
    """Tests for SentenceBuilder.get_sentence()."""

    def test_empty_sentence_returns_empty_string(self):
        builder = SentenceBuilder()
        assert builder.get_sentence() == ""

    def test_words_joined_by_space(self):
        builder = SentenceBuilder(cooldown_seconds=0)
        builder.add_word("Hello")
        builder.add_word("Thank You")
        builder.add_word("Food")
        assert builder.get_sentence() == "Hello Thank You Food"


class TestConfiguration:
    """Tests for SentenceBuilder constructor validation."""

    def test_negative_cooldown_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            SentenceBuilder(cooldown_seconds=-1.0)

    def test_zero_cooldown_is_valid(self):
        builder = SentenceBuilder(cooldown_seconds=0.0)
        assert builder.cooldown_seconds == 0.0
