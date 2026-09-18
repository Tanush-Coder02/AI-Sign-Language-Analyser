"""
test_speech_service.py
----------------------
Unit tests for SpeechService.

The pyttsx3 engine is mocked so these tests run without audio output
or a TTS engine installed.
"""

import pytest

from src.exceptions import SpeechError
from src.speech_service import SpeechService


class TestSpeak:

    def test_speak_calls_engine_say_and_run(self, mocker):
        """speak() should call engine.say() and engine.runAndWait()."""
        mock_engine = mocker.MagicMock()
        mocker.patch("pyttsx3.init", return_value=mock_engine)

        service = SpeechService()
        service.speak("Hello Water Stop")

        mock_engine.say.assert_called_once_with("Hello Water Stop")
        mock_engine.runAndWait.assert_called_once()

    def test_speak_strips_whitespace(self, mocker):
        """Leading/trailing whitespace is stripped before calling TTS."""
        mock_engine = mocker.MagicMock()
        mocker.patch("pyttsx3.init", return_value=mock_engine)

        service = SpeechService()
        service.speak("  Hello  ")

        mock_engine.say.assert_called_once_with("Hello")

    def test_empty_string_raises_speech_error(self):
        service = SpeechService()
        with pytest.raises(SpeechError, match="empty"):
            service.speak("")

    def test_whitespace_only_raises_speech_error(self):
        service = SpeechService()
        with pytest.raises(SpeechError, match="empty"):
            service.speak("   ")

    def test_engine_error_raises_speech_error(self, mocker):
        """If the engine raises during runAndWait, SpeechError is raised."""
        mock_engine = mocker.MagicMock()
        mock_engine.runAndWait.side_effect = RuntimeError("TTS crashed")
        mocker.patch("pyttsx3.init", return_value=mock_engine)

        service = SpeechService()
        with pytest.raises(SpeechError, match="failed while speaking"):
            service.speak("Hello")

    def test_engine_resets_after_failure(self, mocker):
        """After a failure, _engine is reset to None so the next call re-initializes."""
        mock_engine = mocker.MagicMock()
        mock_engine.runAndWait.side_effect = RuntimeError("crash")
        mocker.patch("pyttsx3.init", return_value=mock_engine)

        service = SpeechService()
        with pytest.raises(SpeechError):
            service.speak("Hello")

        assert service._engine is None

    def test_pyttsx3_not_installed_raises_speech_error(self, mocker):
        """If pyttsx3 is not importable, SpeechError is raised."""
        mocker.patch("builtins.__import__", side_effect=_fake_import_error)
        service = SpeechService()
        with pytest.raises(SpeechError, match="not installed"):
            service.speak("Hello")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_import_error(name, *args, **kwargs):
    """Raises ImportError only for pyttsx3, lets everything else through."""
    if name == "pyttsx3":
        raise ImportError("No module named 'pyttsx3'")
    # Fall back to the real import for all other modules
    import builtins
    return builtins.__import__(name, *args, **kwargs)
