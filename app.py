"""
app.py
------
AI Sign Language Analyzer — Streamlit User Interface

This file is the entry point for the application.  It wires together:
  - HandDetector   (MediaPipe hand-landmark extraction)
  - SignClassifier (scikit-learn gesture recognition)
  - StabilityTracker (prevents jitter / accidental acceptances)
  - SentenceBuilder (builds the output sentence word by word)
  - SpeechService  (offline text-to-speech)

HOW TO RUN
----------
  streamlit run app.py

DISCLAIMER
----------
This is a prototype assistive-communication tool.
- It supports only a CUSTOM GESTURE SET (8 signs), NOT ASL, BSL, or any
  other recognized sign language.
- It is NOT a certified medical, emergency, or accessibility device.
- Sign languages are complete, natural languages; this app does NOT
  translate any of them.
- Misrecognition is possible, especially in low-light conditions or with
  hands that differ from the training data.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np
import streamlit as st

# ── Path setup so the app can be launched from the project root ───────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

from src.config import AppConfig
from src.exceptions import (
    AppError,
    CameraError,
    ModelLoadError,
    PredictionError,
    SpeechError,
)
from src.hand_detector import HandDetector
from src.sentence_builder import SentenceBuilder
from src.sign_classifier import SignClassifier
from src.speech_service import SpeechService
from src.stability_tracker import StabilityTracker

# ── File paths ────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(_ROOT, "config.json")
MODEL_PATH = os.path.join(_ROOT, "models", "gesture_model.pkl")
LABELS_PATH = os.path.join(_ROOT, "models", "label_map.json")


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Sign Language Analyzer",
    page_icon="🤟",
    layout="wide",
)


# ═════════════════════════════════════════════════════════════════════════════
# Session-state helpers
# ═════════════════════════════════════════════════════════════════════════════

def _init_session_state(cfg: AppConfig) -> None:
    """Initialize all session-state variables on first load."""
    defaults = {
        "camera_active": False,
        "sentence_builder": SentenceBuilder(cooldown_seconds=cfg.cooldown_seconds),
        "stability_tracker": StabilityTracker(required_frames=cfg.stability_frames),
        "current_sign": "—",
        "current_confidence": 0.0,
        "last_accepted_sign": "—",
        "status_message": "",
        "status_level": "info",   # "info" | "warning" | "error" | "success"
        "model_available": False,
        "classifier": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _set_status(message: str, level: str = "info") -> None:
    st.session_state["status_message"] = message
    st.session_state["status_level"] = level


# ═════════════════════════════════════════════════════════════════════════════
# Model loader (cached so it runs only once per session)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading gesture model…")
def _load_classifier(
    model_path: str, labels_path: str, threshold: float
) -> SignClassifier | None:
    """
    Load the trained model.  Returns None if the model is not yet available,
    allowing the app to start in a degraded (no-prediction) state.
    """
    try:
        return SignClassifier(
            model_path=model_path,
            labels_path=labels_path,
            confidence_threshold=threshold,
        )
    except ModelLoadError as exc:
        return None  # caller checks for None and shows the right message


# ═════════════════════════════════════════════════════════════════════════════
# UI rendering helpers
# ═════════════════════════════════════════════════════════════════════════════

def _render_disclaimer() -> None:
    with st.expander("⚠️ Important Disclaimer — Please Read", expanded=False):
        st.markdown(
            """
**This is a PROTOTYPE assistive-communication tool. It is NOT a certified
medical, emergency, or accessibility device.**

- This application recognizes only a **custom gesture set of 8 signs**.
  It does **not** support American Sign Language (ASL), British Sign Language
  (BSL), or any other recognized sign language.
- Sign languages are complete, complex, natural languages. This app does **not**
  translate any of them.
- Gesture recognition may be inaccurate, especially in poor lighting, with
  different hand sizes, or when the training data does not match your hands.
- **Do not rely on this tool in emergency situations.**
- Collected landmark data is stored **locally only**. No data is sent to any
  external server.
            """
        )


def _render_sidebar(cfg: AppConfig) -> float:
    """
    Render the sidebar with configuration controls.

    Returns the current confidence threshold chosen by the user.
    """
    st.sidebar.header("⚙️ Settings")

    threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=cfg.confidence_threshold,
        step=0.05,
        help=(
            "Minimum prediction confidence to accept a sign. "
            "Lower values accept more signs but may increase errors."
        ),
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Supported Signs")
    for sign in cfg.vocabulary:
        st.sidebar.markdown(f"- **{sign}**")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Custom gesture set only. NOT ASL/BSL or any recognized sign language."
    )

    return threshold


def _render_camera_controls() -> None:
    """Render Start / Stop camera buttons."""
    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            "▶ Start Camera",
            disabled=st.session_state["camera_active"],
            use_container_width=True,
        ):
            st.session_state["camera_active"] = True
            _set_status("Camera started.", "success")
            st.rerun()

    with col2:
        if st.button(
            "⏹ Stop Camera",
            disabled=not st.session_state["camera_active"],
            use_container_width=True,
        ):
            st.session_state["camera_active"] = False
            _set_status("Camera stopped.", "info")
            st.rerun()


def _render_sentence_panel() -> None:
    """Render the sentence display and action buttons."""
    builder: SentenceBuilder = st.session_state["sentence_builder"]
    sentence = builder.get_sentence()

    st.subheader("📝 Current Sentence")
    if sentence:
        st.markdown(
            f"<p style='font-size:1.5rem; font-weight:bold; "
            f"color:#1f2328;'>{sentence}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("(sentence is empty — show signs to the camera to build it)")

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("⬅ Remove Last Word", use_container_width=True):
            removed = builder.remove_last()
            if removed:
                _set_status("Removed last word.", "info")
            else:
                _set_status("Nothing to remove — sentence is already empty.", "warning")
            st.rerun()

    with col2:
        if st.button("🗑 Clear Sentence", use_container_width=True):
            builder.clear()
            _set_status("Sentence cleared.", "info")
            st.rerun()

    with col3:
        if st.button("🔊 Speak Sentence", use_container_width=True):
            if builder.is_empty():
                _set_status(
                    "Cannot speak an empty sentence. Add some signs first.", "warning"
                )
            else:
                _speak_sentence(builder.get_sentence())
            st.rerun()


def _speak_sentence(text: str) -> None:
    """Call SpeechService and update status accordingly."""
    service = SpeechService()
    try:
        service.speak(text)
        _set_status(f'Speaking: "{text}"', "success")
    except SpeechError as exc:
        _set_status(f"Speech error: {exc}", "error")


def _render_status() -> None:
    """Show the current status message at the appropriate severity level."""
    msg = st.session_state.get("status_message", "")
    level = st.session_state.get("status_level", "info")
    if not msg:
        return
    getattr(st, level)(msg)


# ═════════════════════════════════════════════════════════════════════════════
# Webcam loop
# ═════════════════════════════════════════════════════════════════════════════

def _run_camera_loop(
    classifier: SignClassifier | None,
    cfg: AppConfig,
    threshold: float,
) -> None:
    """
    Open the webcam, process frames, and update session state.

    This function runs in a blocking loop until the user stops the camera.
    Streamlit's st.empty() placeholder is used to update the frame display.
    """
    detector = HandDetector(
        min_detection_confidence=0.7,
        max_num_hands=cfg.max_num_hands,
    )
    builder: SentenceBuilder = st.session_state["sentence_builder"]
    tracker: StabilityTracker = st.session_state["stability_tracker"]

    # Streamlit placeholders that we update on every frame
    frame_placeholder = st.empty()
    info_placeholder = st.empty()

    # Camera indicator
    st.markdown(
        "<span style='color:red; font-weight:bold;'>🔴 Camera is ACTIVE</span>",
        unsafe_allow_html=True,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.session_state["camera_active"] = False
        _set_status(
            "Cannot open camera. Check that the webcam is connected and "
            "that permission has been granted.",
            "error",
        )
        detector.close()
        return

    try:
        while st.session_state.get("camera_active", False):
            ret, frame = cap.read()
            if not ret or frame is None:
                raise CameraError("Failed to read frame from camera.")

            # Detect hand and annotate frame
            landmarks, annotated = detector.process_for_display(frame)

            # Flip horizontally for a mirror-like display
            annotated = cv2.flip(annotated, 1)

            # ── Predict ──────────────────────────────────────────────────────
            if landmarks is not None and classifier is not None:
                try:
                    label, confidence = classifier.predict(landmarks)
                    # Update threshold from the sidebar slider in real time
                    classifier.confidence_threshold = threshold
                except (PredictionError, AppError) as exc:
                    label, confidence = "Error", 0.0
                    _set_status(f"Prediction error: {exc}", "error")

                st.session_state["current_sign"] = label
                st.session_state["current_confidence"] = confidence

                # ── Stability check ───────────────────────────────────────
                if label != "Unknown" and label != "Error":
                    is_stable = tracker.update(label)
                    if is_stable:
                        added = builder.add_word(label)
                        if added:
                            tracker.reset()
                            st.session_state["last_accepted_sign"] = label
                            _set_status(f'✅ Added: "{label}"', "success")
                else:
                    tracker.reset()

                # ── Draw sign label on frame ──────────────────────────────
                color = (0, 200, 0) if label != "Unknown" else (0, 100, 255)
                cv2.putText(
                    annotated,
                    f"{label}  {confidence * 100:.0f}%",
                    (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                )

                info_placeholder.markdown(
                    f"**Detected sign:** `{label}` | "
                    f"**Confidence:** `{confidence * 100:.1f}%` | "
                    f"**Last accepted:** `{st.session_state['last_accepted_sign']}`"
                )

            elif landmarks is None:
                tracker.reset()
                st.session_state["current_sign"] = "—"
                st.session_state["current_confidence"] = 0.0
                cv2.putText(
                    annotated,
                    "No hand detected",
                    (10, annotated.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (200, 200, 200),
                    2,
                )
                info_placeholder.info("No hand detected — show your hand to the camera.")

            elif classifier is None:
                info_placeholder.warning(
                    "Model not loaded — hand is visible but cannot classify. "
                    "Run `python scripts/train_model.py` first."
                )

            # ── Display annotated frame ───────────────────────────────────
            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    except CameraError as exc:
        _set_status(f"Camera error: {exc}", "error")
    finally:
        cap.release()
        detector.close()
        st.session_state["camera_active"] = False


# ═════════════════════════════════════════════════════════════════════════════
# Main application entry point
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Load configuration ────────────────────────────────────────────────────
    try:
        cfg = AppConfig.load(CONFIG_PATH)
    except Exception as exc:
        st.error(f"Configuration error: {exc}")
        st.stop()

    # ── Initialize session state ──────────────────────────────────────────────
    _init_session_state(cfg)

    # ── Page title and disclaimer ─────────────────────────────────────────────
    st.title("🤟 AI Sign Language Analyzer")
    st.caption(
        "Prototype assistive-communication tool | Custom gesture set | "
        "NOT a certified medical or emergency device"
    )
    _render_disclaimer()

    # ── Sidebar (settings + vocabulary list) ──────────────────────────────────
    threshold = _render_sidebar(cfg)

    # ── Load or retrieve the classifier ──────────────────────────────────────
    classifier = _load_classifier(MODEL_PATH, LABELS_PATH, threshold)

    if classifier is None:
        st.warning(
            "⚠️ **No trained model found.** "
            "The application will show the camera feed and detect hands, "
            "but cannot classify signs.\n\n"
            "To train the model:\n"
            "1. `python scripts/collect_data.py --label Hello --count 150`\n"
            "   (repeat for each sign)\n"
            "2. `python scripts/train_model.py`"
        )

    # ── Camera controls ───────────────────────────────────────────────────────
    st.subheader("📷 Camera")
    _render_camera_controls()

    # ── Status message ────────────────────────────────────────────────────────
    _render_status()

    # ── Sentence panel ────────────────────────────────────────────────────────
    st.markdown("---")
    _render_sentence_panel()

    # ── Webcam loop (runs only when camera is active) ─────────────────────────
    if st.session_state.get("camera_active", False):
        st.markdown("---")
        _run_camera_loop(classifier, cfg, threshold)
        # After the loop exits (user pressed Stop or error), rerun to refresh UI
        st.rerun()

    else:
        st.markdown("---")
        st.info("Camera is off. Press **▶ Start Camera** to begin.")


if __name__ == "__main__":
    main()
