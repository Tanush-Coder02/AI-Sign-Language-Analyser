"""
train_model.py
--------------
Trains a RandomForest (or SVM) classifier on the collected gesture samples,
evaluates it on a held-out test set, and saves the model + label map.

Usage
-----
  # Train with default settings (RandomForest):
  python scripts/train_model.py

  # Use SVM instead:
  python scripts/train_model.py --model svm

  # Specify a different data file:
  python scripts/train_model.py --data data/gesture_samples.csv

Output
------
  models/gesture_model.pkl  — the trained classifier
  models/label_map.json     — {0: "Hello", 1: "Yes", ...}

The script will NOT use the test split for training, and reports accuracy
plus a full classification report on the unseen test set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import joblib
import numpy as np

# Allow imports from the project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "gesture_samples.csv")
MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")
LABELS_PATH = os.path.join(MODELS_DIR, "label_map.json")

MIN_SAMPLES_PER_CLASS = 20  # warn if any class has fewer samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a hand-gesture classifier from collected samples."
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the CSV file with gesture samples (default: {DEFAULT_DATA_PATH}).",
    )
    parser.add_argument(
        "--model",
        choices=["rf", "svm"],
        default="rf",
        help="Classifier to train: 'rf' (RandomForest) or 'svm' (SVM). Default: rf.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to reserve for testing (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    return parser.parse_args()


def load_data(data_path: str):
    """
    Load gesture_samples.csv and return features (X) and labels (y).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, dict]
        (X, y_encoded, label_map)
        where y_encoded is an integer array and label_map maps int → str.
    """
    import pandas as pd  # imported here to avoid hard dependency at module level

    if not os.path.isfile(data_path):
        print(f"ERROR: Data file not found: '{data_path}'")
        print("Run 'python scripts/collect_data.py --label <Sign> --count 150' first.")
        sys.exit(1)

    df = pd.read_csv(data_path)

    if df.empty:
        print("ERROR: The data file is empty. Collect some samples first.")
        sys.exit(1)

    if "label" not in df.columns:
        print("ERROR: The CSV file must have a 'label' column.")
        sys.exit(1)

    # Split features and labels
    X = df.drop(columns=["label"]).values.astype(np.float32)
    raw_labels = df["label"].values

    # Encode string labels to integers
    unique_labels = sorted(set(raw_labels))
    label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
    label_map = {idx: label for label, idx in label_to_int.items()}

    y = np.array([label_to_int[label] for label in raw_labels], dtype=np.int32)

    print(f"\nLoaded {len(X)} samples across {len(unique_labels)} classes.")
    print(f"Classes: {unique_labels}\n")

    # Warn about classes with few samples
    for label in unique_labels:
        count = int(np.sum(raw_labels == label))
        if count < MIN_SAMPLES_PER_CLASS:
            print(
                f"WARNING: '{label}' has only {count} sample(s). "
                f"Aim for at least {MIN_SAMPLES_PER_CLASS} for reliable training."
            )

    return X, y, label_map


def build_classifier(model_type: str, random_state: int):
    """Return an untrained scikit-learn classifier."""
    if model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "svm":
        from sklearn.svm import SVC
        return SVC(
            kernel="rbf",
            probability=True,  # required for predict_proba()
            random_state=random_state,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type!r}")


def train(args: argparse.Namespace) -> None:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    # ── Load data ─────────────────────────────────────────────────────────────
    X, y, label_map = load_data(args.data)

    if len(set(y)) < 2:
        print("ERROR: Need at least 2 classes to train. Collect more labels.")
        sys.exit(1)

    # ── Split — stratified so every class appears in both sets ───────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"Training samples : {len(X_train)}")
    print(f"Test samples     : {len(X_test)}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    clf = build_classifier(args.model, args.random_state)
    print(f"Training {clf.__class__.__name__}…")
    clf.fit(X_train, y_train)
    print("Training complete.\n")

    # ── Evaluate on the HELD-OUT test set (never used for training) ───────────
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    target_names = [label_map[i] for i in sorted(label_map.keys())]
    print(f"Test Accuracy : {accuracy * 100:.2f}%\n")
    print("Classification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        )
    )

    print("Confusion Matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred)
    # Print with label headers for readability
    header = "       " + "  ".join(f"{n[:6]:>6}" for n in target_names)
    print(header)
    for row_label, row in zip(target_names, cm):
        print(f"{row_label[:7]:>7}  " + "  ".join(f"{v:>6}" for v in row))

    # ── Save model and label map ───────────────────────────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    with open(LABELS_PATH, "w", encoding="utf-8") as fh:
        json.dump(label_map, fh, indent=2)

    print(f"\nModel saved  : {MODEL_PATH}")
    print(f"Label map    : {LABELS_PATH}")
    print("\nTraining finished successfully.")


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
