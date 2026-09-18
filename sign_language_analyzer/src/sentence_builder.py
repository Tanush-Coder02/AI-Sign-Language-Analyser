"""
sentence_builder.py
-------------------
Manages the sentence that the user is building word by word.

Responsibilities:
- Keep a list of accepted words.
- Prevent adding the same word twice within a configurable cooldown window.
- Allow removing the last word.
- Allow clearing the entire sentence.
- Return the sentence as a single string.
"""

from __future__ import annotations

import time
from typing import List


class SentenceBuilder:
    """
    Builds a sentence by accumulating accepted sign words.

    Parameters
    ----------
    cooldown_seconds : float
        Minimum time that must pass before the same sign can be added again.
        Setting this to 0 disables duplicate prevention entirely.

    Example
    -------
    builder = SentenceBuilder(cooldown_seconds=2.0)
    builder.add_word("Hello")   # returns True  → sentence: "Hello"
    builder.add_word("Hello")   # returns False (cooldown) → sentence: "Hello"
    builder.add_word("Water")   # returns True  → sentence: "Hello Water"
    builder.remove_last()       #               → sentence: "Hello"
    builder.get_sentence()      # returns "Hello"
    """

    def __init__(self, cooldown_seconds: float = 2.0) -> None:
        if cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be non-negative. Got: {cooldown_seconds!r}"
            )
        self.cooldown_seconds = cooldown_seconds
        self._words: List[str] = []
        self._last_sign: str = ""
        self._last_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def add_word(self, word: str) -> bool:
        """
        Attempt to add a word to the sentence.

        The word is rejected (returns False) if:
        - It is empty or whitespace-only.
        - It equals the last added word AND the cooldown has not expired.

        Parameters
        ----------
        word : str
            The sign label to add (e.g. "Hello").

        Returns
        -------
        bool
            True if the word was added; False if it was rejected.
        """
        # Reject empty or whitespace-only words
        word = word.strip()
        if not word:
            return False

        # Reject Unknown sign
        if word.lower() == "unknown":
            return False

        # Cooldown check: same word too soon
        now = time.time()
        if (
            word == self._last_sign
            and (now - self._last_time) < self.cooldown_seconds
        ):
            return False

        self._words.append(word)
        self._last_sign = word
        self._last_time = now
        return True

    def remove_last(self) -> bool:
        """
        Remove the most recently added word.

        Returns
        -------
        bool
            True if a word was removed; False if the sentence was already empty.
        """
        if not self._words:
            return False
        self._words.pop()
        # Reset last-sign tracking so the same word can be added immediately
        self._last_sign = ""
        self._last_time = 0.0
        return True

    def clear(self) -> None:
        """Remove all words and reset tracking state."""
        self._words.clear()
        self._last_sign = ""
        self._last_time = 0.0

    def get_sentence(self) -> str:
        """
        Return all words joined by spaces.

        Returns
        -------
        str
            e.g. "Hello Water Food" or "" if no words have been added.
        """
        return " ".join(self._words)

    def is_empty(self) -> bool:
        """Return True if no words have been added yet."""
        return len(self._words) == 0

    @property
    def words(self) -> List[str]:
        """Read-only view of the current word list."""
        return list(self._words)
