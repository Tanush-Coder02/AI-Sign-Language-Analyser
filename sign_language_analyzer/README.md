# 🤟 AI Sign Language Analyzer

A beginner-friendly, prototype assistive-communication tool that uses a
webcam to recognize a small set of predefined hand gestures and converts
them into readable text and speech.

> **Important:** This application supports only a **custom gesture set of
> 8 signs**. It does **not** support American Sign Language (ASL), British
> Sign Language (BSL), or any other recognized sign language. Sign languages
> are complete, natural languages. This is a prototype — not a certified
> medical, emergency, or accessibility device.

---

## Supported Signs

| Sign | Suggested Gesture |
|---|---|
| Hello | Open palm facing camera, fingers together |
| Yes | Closed fist |
| No | Index and middle finger extended, spread apart |
| Help | Thumbs up |
| Thank You | Flat palm facing camera |
| Water | Three middle fingers extended (W shape) |
| Food | Fingers pinched together |
| Stop | Open palm, fingers spread wide |

You can redefine these gestures during data collection — just be consistent.

---

## Technology Stack

| Component | Technology |
|---|---|
| User Interface | Streamlit |
| Camera Input | OpenCV |
| Hand Detection | MediaPipe Hands |
| Gesture Classification | scikit-learn (RandomForest or SVM) |
| Text-to-Speech | pyttsx3 (offline) |
| Testing | pytest + pytest-mock |
| Data Storage | CSV (training data), JSON (config + label map) |

---

## Project Structure

```
sign_language_analyzer/
├── app.py                     # Streamlit entry point
├── config.json                # Runtime configuration
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py              # AppConfig dataclass + loader
│   ├── exceptions.py          # Custom exceptions
│   ├── hand_detector.py       # HandDetector (MediaPipe wrapper)
│   ├── sign_classifier.py     # SignClassifier (sklearn wrapper)
│   ├── sentence_builder.py    # SentenceBuilder
│   ├── speech_service.py      # SpeechService (pyttsx3 wrapper)
│   └── stability_tracker.py  # StabilityTracker (debounce helper)
│
├── scripts/
│   ├── collect_data.py        # CLI: capture labelled gesture samples
│   └── train_model.py         # CLI: train, evaluate, save model
│
├── models/
│   ├── gesture_model.pkl      # Saved trained classifier (created after training)
│   └── label_map.json         # {0: "Hello", 1: "Yes", ...} (created after training)
│
├── data/
│   └── gesture_samples.csv    # Collected training data (created after collection)
│
└── tests/
    ├── test_sentence_builder.py
    ├── test_stability_tracker.py
    ├── test_sign_classifier.py
    ├── test_speech_service.py
    └── test_config.py
```

---

## Installation

### Prerequisites

- Python 3.10 or newer
- A working webcam
- Windows, macOS, or Linux

### 1. Clone or download the project

```bash
git clone <repository-url>
cd sign_language_analyzer
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Linux only — install espeak for text-to-speech

```bash
sudo apt-get install espeak
```

---

## How to Collect Gesture Data

Before training the model you need to collect hand-landmark samples for
each sign. No images are saved — only numeric coordinate data.

```bash
# Collect 150 samples for "Hello"
python scripts/collect_data.py --label Hello --count 150

# Collect with auto-capture (captures automatically when a hand is detected)
python scripts/collect_data.py --label Yes --count 150 --auto

# Repeat for each sign in the vocabulary
python scripts/collect_data.py --label No        --count 150 --auto
python scripts/collect_data.py --label Help      --count 150 --auto
python scripts/collect_data.py --label "Thank You" --count 150 --auto
python scripts/collect_data.py --label Water     --count 150 --auto
python scripts/collect_data.py --label Food      --count 150 --auto
python scripts/collect_data.py --label Stop      --count 150 --auto
```

**Controls during collection:**
- `SPACE` — capture one sample (manual mode)
- `A` — toggle auto-capture on / off
- `Q` — quit

**To delete samples for a label:**

```bash
python scripts/collect_data.py --label Hello --delete
```

> **Privacy note:** Only landmark coordinates (42 numbers per sample) are
> saved to `data/gesture_samples.csv`. No images or video are recorded.

---

## How to Train the Model

Once you have collected samples for all signs, run:

```bash
# Train a Random Forest (default, recommended for beginners)
python scripts/train_model.py

# Alternatively, train an SVM
python scripts/train_model.py --model svm
```

The script will print accuracy and a classification report, then save:
- `models/gesture_model.pkl` — the trained classifier
- `models/label_map.json` — the label mapping

**Example output:**
```
Loaded 1200 samples across 8 classes.

Training samples : 960
Test samples     : 240

Training RandomForestClassifier…
Training complete.

Test Accuracy : 97.50%

Classification Report:
              precision    recall  f1-score   support
       Hello       0.98      0.99      0.99        30
         Yes       0.97      0.97      0.97        30
          No       0.98      0.97      0.97        30
        Help       1.00      1.00      1.00        30
    Thank You       0.97      0.97      0.97        30
       Water       0.97      0.97      0.97        30
        Food       0.97      0.97      0.97        30
        Stop       0.97      0.97      0.97        30
```

---

## How to Run Tests

```bash
# Run all unit tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_sentence_builder.py -v

# Run with test coverage (requires pytest-cov)
pip install pytest-cov
pytest tests/ --cov=src --cov-report=term-missing
```

All tests pass without a physical camera, model file, or TTS engine.

---

## How to Start the Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

**First-time workflow:**
1. Read the disclaimer.
2. Press **▶ Start Camera**.
3. Show a sign to the camera.
4. Wait for the sign to be accepted (stability check requires ~10 steady frames).
5. The word appears in the sentence panel.
6. Build your sentence, then press **🔊 Speak Sentence**.

---

## Configuration

Edit `config.json` to change default settings:

```json
{
    "confidence_threshold": 0.75,
    "stability_frames": 10,
    "cooldown_seconds": 2.0,
    "max_num_hands": 1,
    "vocabulary": ["Hello", "Yes", "No", "Help", "Thank You", "Water", "Food", "Stop"]
}
```

| Setting | Description | Default |
|---|---|---|
| `confidence_threshold` | Minimum confidence (0–1) to accept a prediction | `0.75` |
| `stability_frames` | Frames the sign must be held steadily | `10` |
| `cooldown_seconds` | Seconds before the same sign can be added again | `2.0` |
| `max_num_hands` | Number of hands to detect (1 recommended) | `1` |
| `vocabulary` | List of accepted sign labels | 8 signs |

The **Confidence Threshold** can also be adjusted live in the sidebar slider.

---

## Privacy Information

- **No images or video are saved** by default.
- Data collection saves only numeric landmark coordinates (`data/gesture_samples.csv`).
- No data is sent to any external server — everything runs locally.
- To delete collected training data for a sign:
  ```bash
  python scripts/collect_data.py --label Hello --delete
  ```
- To delete all training data, delete `data/gesture_samples.csv`.

---

## Known Limitations

- **Static gestures only:** Signs that require movement (e.g., waving) cannot
  be recognized in this MVP. Only hand poses are classified.
- **Single-user training:** A model trained on one person's hand will not
  generalize perfectly to other people's hands, lighting, or skin tones.
- **One hand only:** The app tracks only the first detected hand.
- **No real sign languages:** This is a custom gesture set, not ASL, BSL, etc.
- **TTS quality varies by OS:** Windows (SAPI5), macOS (NSSpeechSynthesizer),
  and Linux (espeak) have different voice quality.
- **Streamlit webcam loop:** The webcam loop blocks the Streamlit server thread;
  other UI interactions are paused while the camera is active.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Cannot open camera" | Check webcam connection and OS camera permissions |
| "No trained model found" | Run `python scripts/train_model.py` after collecting data |
| "label_map.json not found" | Re-run `python scripts/train_model.py` |
| Low accuracy | Collect more samples (aim for 200+); ensure consistent lighting |
| TTS not working on Linux | Install espeak: `sudo apt-get install espeak` |
| Import errors | Ensure virtual environment is activated and `pip install -r requirements.txt` was run |
| Signs detected but never accepted | Lower the confidence threshold slider; check lighting |

---

## Future Improvement Ideas

- **Motion gestures:** Add LSTM or Transformer model for temporal sequences.
- **Two-hand signs:** Process both hands simultaneously.
- **User-defined vocabulary:** Let users record and label their own gestures in the UI.
- **Real ASL/BSL support:** Use a large public dataset (requires significant data and a larger model).
- **Export sentence:** Copy to clipboard or save to a text file.
- **Mobile deployment:** Explore browser-based MediaPipe for mobile use.
- **Multi-user training:** Collect data from multiple users to improve fairness.

---

## Disclaimer

This application is a **beginner-friendly prototype** built for educational
and assistive-communication experimentation.

- It is **NOT** a certified, validated, or regulated accessibility device.
- It is **NOT** suitable for emergency, medical, or safety-critical communication.
- Sign languages (ASL, BSL, ISL, etc.) are complete natural languages. This
  application recognizes only a small custom gesture set and **does not claim
  to understand any sign language**.
- Misrecognition is always possible. Do not rely on this tool where accurate
  communication is critical.
