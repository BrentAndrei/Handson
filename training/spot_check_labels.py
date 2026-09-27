"""
spot_check_labels.py

Diagnostic script for investigating possible mislabeling in FSL-105 classes.

PART 1 — Thumbnails:
For each of 8 labels (5 targets + 3 controls), extracts the middle frame
from 3 clips and saves to training/spot_check_thumbnails/<LABEL>/<n>.jpg.

PART 2 — Consistency check:
For each clip, computes a single 345-dim summary vector (mean of per-frame
select_features() across all frames), then computes average pairwise
Euclidean distance between the 3 clips within each label.

Usage (from handson/training/):
    .venv-extract\Scripts\activate
    python spot_check_labels.py
"""

import csv
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# --- Import shared landmarker utilities from extract_landmarks.py ------------
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from extract_landmarks import (
    POSE_COUNT,
    HAND_COUNT,
    FACE_COUNT,
    POSE_OFFSET,
    LEFT_HAND_OFFSET,
    RIGHT_HAND_OFFSET,
    FACE_OFFSET,
    TOTAL_FLOATS,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LIP_LANDMARK_INDICES,
    FEATURE_DIM,
    TARGET_FPS,
    ensure_model,
    make_landmarker,
    write_landmarks,
    normalize,
    select_features,
    get_landmark_group,
)


# --- Config ------------------------------------------------------------------
FSL105_ROOT = Path(r"C:\Users\BRENT\Downloads\FSL-105\FSL-105 A dataset for recognizing 105 Filipino sign language videos")
LABELS_CSV = FSL105_ROOT / "labels.csv"
TRAIN_CSV = FSL105_ROOT / "train.csv"

TARGET_LABELS = ["GRANDMOTHER", "FOUR", "GREEN", "KNOW", "COLD"]
CONTROL_LABELS = ["HELLO", "THANK YOU", "GOOD MORNING"]
ALL_LABELS = TARGET_LABELS + CONTROL_LABELS

OUTPUT_DIR = SCRIPT_DIR / "spot_check_thumbnails"


def load_label_ids():
    id_to_name = {}
    with open(LABELS_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_name[row["id"].strip()] = row["label"].strip()
    return id_to_name


def get_clips_for_labels(id_to_name):
    name_to_id = {v: k for k, v in id_to_name.items()}
    target_ids = {name: name_to_id[name] for name in ALL_LABELS if name in name_to_id}

    paths = {name: [] for name in ALL_LABELS}
    with open(TRAIN_CSV, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lid = row["id_label"].strip()
            for name in ALL_LABELS:
                if name in target_ids and target_ids[name] == lid:
                    paths[name].append(row["vid_path"].strip())
    return paths


def extract_thumbnail(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path}")
        return False
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return False
    mid = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return False
    cv2.imwrite(output_path, frame)
    return True


def run_thumbnails(paths):
    print("=== PART 1: Extracting thumbnails ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label in ALL_LABELS:
        out_dir = OUTPUT_DIR / label
        out_dir.mkdir(parents=True, exist_ok=True)
        vids = paths.get(label, [])[:3]
        for i, vid_rel in enumerate(vids):
            vid_full = FSL105_ROOT / vid_rel
            out_path = str(out_dir / f"{i}.jpg")
            ok = extract_thumbnail(str(vid_full), out_path)
            status = "OK" if ok else "FAILED"
            print(f"  {label}: {vid_rel} -> {out_path} [{status}]")


def compute_clip_summary(video_path: str) -> np.ndarray:
    landmarker = make_landmarker()
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
        duration_sec = frame_count / native_fps
        target_frame_count = max(1, round(duration_sec * TARGET_FPS))
        sample_indices = np.linspace(0, frame_count - 1, target_frame_count).astype(int)
        sample_indices = np.unique(sample_indices)

        features = []
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int(len(features) * 1000 / TARGET_FPS)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            raw = np.zeros(TOTAL_FLOATS, dtype=np.float32)
            has_pose = write_landmarks(
                raw, POSE_OFFSET, POSE_COUNT, get_landmark_group(result.pose_landmarks)
            )
            write_landmarks(
                raw, LEFT_HAND_OFFSET, HAND_COUNT,
                get_landmark_group(result.left_hand_landmarks),
            )
            write_landmarks(
                raw, RIGHT_HAND_OFFSET, HAND_COUNT,
                get_landmark_group(result.right_hand_landmarks),
            )
            write_landmarks(
                raw, FACE_OFFSET, FACE_COUNT, get_landmark_group(result.face_landmarks)
            )
            normalized = normalize(raw, has_pose)
            features.append(select_features(normalized))

        cap.release()
        if not features:
            return np.zeros(FEATURE_DIM, dtype=np.float32)
        return np.mean(np.stack(features), axis=0)
    finally:
        landmarker.close()


def pairwise_distances(vectors):
    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            d = np.linalg.norm(vectors[i] - vectors[j])
            dists.append(d)
    return dists


def run_consistency_check(paths):
    print("\n=== PART 2: Consistency check ===")
    results = {}
    for label in ALL_LABELS:
        vids = paths.get(label, [])[:3]
        vecs = []
        for vid_rel in vids:
            vid_full = FSL105_ROOT / vid_rel
            print(f"  Processing {label}: {vid_rel}")
            vec = compute_clip_summary(str(vid_full))
            vecs.append(vec)
        results[label] = vecs

    print("\nLabel            Avg intra-label distance")
    print("-" * 45)
    stats = []
    for label in ALL_LABELS:
        vecs = results[label]
        dists = pairwise_distances(vecs)
        avg = np.mean(dists) if dists else 0.0
        stats.append((label, avg))
    stats.sort(key=lambda x: x[1])
    for label, avg in stats:
        print(f"{label:<20} {avg:>12.4f}")


def main():
    print("Loading FSL-105 metadata...")
    id_to_name = load_label_ids()
    paths = get_clips_for_labels(id_to_name)

    run_thumbnails(paths)
    run_consistency_check(paths)


if __name__ == "__main__":
    main()
