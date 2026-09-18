"""
collect_data.py
---------------
Command-line script for collecting labelled hand-gesture training samples.

IMPORTANT PRIVACY NOTICE
-------------------------
This script saves ONLY numeric landmark coordinates (42 floats per sample)
to data/gesture_samples.csv.  NO images or video are saved to disk.

Usage
-----
  # Collect 150 samples for the "Hello" sign:
  python scripts/collect_data.py --label Hello --count 150

  # Auto-capture mode (captures continuously when hand is detected):
  python scripts/collect_data.py --label Yes --count 100 --auto

  # Delete all samples for a label:
  python scripts/collect_data.py --label Hello --delete

Controls (manual mode)
----------------------
  SPACE  — capture one sample
  A      — toggle auto-capture on/off
  Q      — quit

The script appends to the CSV file, so you can collect multiple sessions.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2

# Allow imports from the project root regardless of where the script is run
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from src.hand_detector import HandDetector  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")
CSV_PATH = os.path.join(DATA_DIR, "gesture_samples.csv")
NUM_LANDMARKS = 42  # 21 keypoints × 2 (x, y)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect labelled hand-gesture landmark samples."
    )
    parser.add_argument(
        "--label",
        required=True,
        help="The sign label to record (e.g. Hello, Yes, Stop).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=150,
        help="Number of samples to collect (default: 150).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Start in auto-capture mode.",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete all existing samples for --label and exit.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (default: 0).",
    )
    return parser.parse_args()


def delete_samples(label: str) -> None:
    """Remove all rows for `label` from the CSV file."""
    if not os.path.isfile(CSV_PATH):
        print(f"No data file found at {CSV_PATH}. Nothing to delete.")
        return

    rows_kept = []
    rows_deleted = 0
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        for row in reader:
            if row and row[0] == label:
                rows_deleted += 1
            else:
                rows_kept.append(row)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if header:
            writer.writerow(header)
        writer.writerows(rows_kept)

    print(f"Deleted {rows_deleted} sample(s) for label '{label}'.")


def ensure_csv_header() -> None:
    """Write the CSV header row if the file does not exist yet."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.isfile(CSV_PATH):
        header = ["label"] + [f"lm_{i}" for i in range(NUM_LANDMARKS)]
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(header)


def count_existing(label: str) -> int:
    """Return how many samples already exist for this label."""
    if not os.path.isfile(CSV_PATH):
        return 0
    count = 0
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header
        for row in reader:
            if row and row[0] == label:
                count += 1
    return count


def collect(label: str, target_count: int, auto_start: bool, camera_idx: int) -> None:
    """Run the interactive data-collection loop."""
    ensure_csv_header()
    existing = count_existing(label)
    collected = 0
    needed = target_count

    print(f"\n{'=' * 55}")
    print(f"  Collecting samples for label: '{label}'")
    print(f"  Existing samples: {existing}")
    print(f"  Target new samples: {needed}")
    print(f"  Controls: SPACE=capture  A=auto-toggle  Q=quit")
    print(f"{'=' * 55}\n")

    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camera_idx}. Check device connection.")
        sys.exit(1)

    detector = HandDetector(min_detection_confidence=0.7, max_num_hands=1)
    auto_capture = auto_start
    last_capture_time = 0.0
    AUTO_INTERVAL = 0.1  # seconds between auto-captures

    with open(CSV_PATH, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)

        while collected < needed:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("ERROR: Cannot read frame from camera.")
                break

            # Detect hand and get annotated frame
            landmarks, annotated = detector.process_for_display(frame)

            hand_msg = "Hand detected" if landmarks else "No hand detected"
            mode_msg = "AUTO" if auto_capture else "MANUAL"

            # Draw status text on the frame
            cv2.putText(
                annotated,
                f"Label: {label}  |  {collected}/{needed}  |  {mode_msg}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0) if landmarks else (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated,
                hand_msg,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0) if landmarks else (0, 0, 255),
                2,
            )

            cv2.imshow("Collect Gesture Data — Press Q to quit", annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting data collection.")
                break
            elif key == ord("a"):
                auto_capture = not auto_capture
                print(f"Auto-capture {'ON' if auto_capture else 'OFF'}")
            elif key == ord(" ") and landmarks:
                # Manual capture
                writer.writerow([label] + landmarks)
                collected += 1
                print(f"  Captured {collected}/{needed}", end="\r")

            # Auto-capture
            if auto_capture and landmarks:
                now = time.time()
                if now - last_capture_time >= AUTO_INTERVAL:
                    writer.writerow([label] + landmarks)
                    collected += 1
                    last_capture_time = now
                    print(f"  Auto-captured {collected}/{needed}", end="\r")

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

    print(f"\nDone. Collected {collected} new sample(s) for '{label}'.")
    print(f"Total samples for '{label}': {existing + collected}")


def main() -> None:
    args = parse_args()

    if args.delete:
        delete_samples(args.label)
        return

    collect(
        label=args.label,
        target_count=args.count,
        auto_start=args.auto,
        camera_idx=args.camera,
    )


if __name__ == "__main__":
    main()
