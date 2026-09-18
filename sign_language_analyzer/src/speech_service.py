"""
speech_service.py
-----------------
Wraps the pyttsx3 offline text-to-speech engine.

pyttsx3 works without an internet connection and uses the OS native TTS:
- Windows: SAPI5
- macOS:   NSSpeechSynthesizer
- Linux:   espeak (must be installed separately)

If the engine cannot be initialized or fails to speak, a SpeechError is
raised so the calling code can show a meaningful message in the UI.
"""

from __future__ import annotations

from src.exceptions import SpeechError


class SpeechService:
    """
    Converts a text string to speech using the offline pyttsx3 engine.

    The engine is initialized lazily (on the first call to speak()) to
    avoid errors at import time on systems where TTS is not available.
    """

    def __init__(self) -> None:
        # _engine stays None until speak() is called the first time.
        self._engine = None

    # ── Public API ────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """
        Speak the given text aloud using the OS text-to-speech engine.

        Parameters
        ----------
        text : str
            The sentence to read aloud.  Must be non-empty.

        Raises
        ------
        SpeechError
            If text is empty, if the TTS engine cannot be initialized,
            or if the engine raises an error while speaking.
        """
        text = text.strip()
        if not text:
            raise SpeechError("Cannot speak an empty sentence.")

        engine = self._get_engine()

        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:
            # The engine may be in a bad state; reset it so next call retries
            self._engine = None
            raise SpeechError(
                f"Text-to-speech failed while speaking: {exc}"
            ) from exc

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_engine(self):
        """
        Return the pyttsx3 engine, initializing it on first use.

        Raises
        ------
        SpeechError
            If pyttsx3 is not installed or cannot find a TTS driver.
        """
        if self._engine is not None:
            return self._engine

        try:
            import pyttsx3  # imported here to keep the module importable even
                             # if pyttsx3 is not installed (for testing)
            self._engine = pyttsx3.init()
        except ImportError as exc:
            raise SpeechError(
                "pyttsx3 is not installed. Run: pip install pyttsx3"
            ) from exc
        except Exception as exc:
            raise SpeechError(
                f"Could not initialize the text-to-speech engine: {exc}\n"
                "On Linux, install espeak: sudo apt-get install espeak"
            ) from exc

        return self._engine
