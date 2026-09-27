"""
Extracts landmark windows from the Roboflow "Filipino Sign Language
Recognition" COCO-format export (research-xbc14/filipino-sign-language-
recognition). Unlike the Kaggle alphabet set, these images are frames
sampled from real video clips of each sign being performed — filenames
like "16-YW_MOV-6_jpg.rf.<hash>.jpg" encode a clip id ("16-YW_MOV") and a
frame index (6) — so this script regroups frames back into per-clip
sequences and runs them through HolisticLandmarker the same way
extract_landmarks.py processes real video files, rather than treating each
image as an independent static pose the way extract_landmarks_alphabet.py
does for fingerspelled letters.

Label source: each image's class comes from the COCO "annotations" array
(image_id -> category_id -> category name), NOT parsed from the filename —
filename abbreviations (YW, GA, ILY, ...) are inconsistent and unreliable.
category_id 0 ("filipino-sign-language-words") is Roboflow's auto-added
root/supercategory placeholder, not a real class, and is skipped.

Label mapping: 9 of these 14 signs are the same words you likely already
have from FSL-105 (just formatted differently — "hello" vs "HELLO", "i-m
fine" vs your existing "IM FINE"). ROBOFLOW_LABEL_MAP below normalizes
each raw Roboflow name to match your existing label spelling exactly for
the overlapping ones, so this data adds MORE training samples to signs
your model already knows, rather than splintering them into separate
near-duplicate classes. Double check this map against your actual
label_map.json before merging — it was built by hand against the labels
reported earlier in this project's chat history, not verified live.

Setup:
    pip install -r requirements-extract.txt

Usage:
    python extract_landmarks_roboflow.py --root path\to\roboflow_fsl_phrases --out roboflow_dataset.json

    This processes train/, valid/, and test/ subfolders together (each
    must contain its own _annotations.coco.json alongside the images —
    that's how Roboflow's COCO export is structured).
"""

import argparse
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# --- Mirrors mediapipe/types.ts (identical to extract_landmarks.py) -------
POSE_COUNT = 33
HAND_COUNT = 21
FACE_COUNT = 468
POSE_OFFSET = 0
LEFT_HAND_OFFSET = POSE_OFFSET + POSE_COUNT * 3
RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + HAND_COUNT * 3
FACE_OFFSET = RIGHT_HAND_OFFSET + HAND_COUNT * 3
TOTAL_FLOATS = FACE_OFFSET + FACE_COUNT * 3

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# --- Mirrors features/featureSelect.ts -------------------------------------
LIP_LANDMARK_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311,
    312, 13, 82, 81, 42, 183, 78,
]
FEATURE_DIM = (POSE_COUNT + HAND_COUNT * 2 + len(LIP_LANDMARK_INDICES)) * 3

# --- Mirrors features/windowBuffer.ts --------------------------------------
WINDOW_SIZE = 16
DEFAULT_STRIDE = 6
TARGET_FPS = 24

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
MODEL_CACHE_PATH = Path(__file__).parent / "holistic_landmarker.task"

# Raw Roboflow category name -> canonical label to train with. Left side
# must match categories[].name in the COCO json exactly. Right side: for
# the 9 overlapping signs, this MUST match your existing label_map.json
# spelling exactly or you'll get duplicate near-identical classes instead
# of more samples for one class — verify against your actual
# label_map.json before running the real merge.
ROBOFLOW_LABEL_MAP = {
    "good-afternoon": "GOOD AFTERNOON",
    "good-evening": "GOOD EVENING",
    "good-morning": "GOOD MORNING",
    "goodbye": "GOODBYE",                    # new
    "hello": "HELLO",
    "how-are-you": "HOW ARE YOU",
    "i-love-you": "I LOVE YOU",               # new
    "i-m fine": "IM FINE",
    "nice to meet you": "NICE TO MEET YOU",
    "please": "PLEASE",                       # new
    "sorry": "SORRY",                         # new
    "thank-you": "THANK YOU",
    "what-s your name": "WHAT'S YOUR NAME",   # new
    "youre-welcome": "YOURE WELCOME",
}

# Matches "<clip>-<frame>_jpg.rf.<hash>.jpg", e.g.
# "16-YW_MOV-6_jpg.rf.0028c53a90c16fb1298883617255e17c.jpg"
# -> clip="16-YW_MOV", frame=6
FILENAME_RE = re.compile(r"^(?P<clip>.+)-(?P<frame>\d+)_jpg\.rf\.[0-9a-f]+\.(jpg|jpeg|png)$", re.IGNORECASE)


def ensure_model() -> str:
    if not MODEL_CACHE_PATH.exists():
        print(f"Downloading holistic landmarker model to {MODEL_CACHE_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE_PATH)
    return str(MODEL_CACHE_PATH)


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=ensure_model()),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_landmarks_confidence=0.3,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.3,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def write_landmarks(target: np.ndarray, offset: int, count: int, landmarks) -> bool:
    if not landmarks or len(landmarks) == 0:
        target[offset : offset + count * 3] = 0
        return False
    for i in range(count):
        p = landmarks[i]
        base = offset + i * 3
        target[base] = p.x
        target[base + 1] = p.y
        target[base + 2] = p.z
    return True


def normalize(vector: np.ndarray, has_pose: bool) -> np.ndarray:
    out = vector.copy()

    def point(group_offset, idx):
        base = group_offset + idx * 3
        return vector[base : base + 3]

    if has_pose:
        ls, rs = point(POSE_OFFSET, LEFT_SHOULDER), point(POSE_OFFSET, RIGHT_SHOULDER)
        anchor = (ls + rs) / 2
        shoulder_dist = float(np.linalg.norm(ls - rs))
        scale = shoulder_dist if shoulder_dist > 1e-4 else bbox_diagonal(vector)
    else:
        anchor = centroid(vector)
        scale = bbox_diagonal(vector)

    if scale <= 1e-4:
        scale = 1.0

    points = out.reshape(-1, 3)
    points[:] = (points - anchor) / scale
    return out


def centroid(vector: np.ndarray) -> np.ndarray:
    points = vector.reshape(-1, 3)
    nonzero = points[np.any(points != 0, axis=1)]
    if len(nonzero) == 0:
        return np.zeros(3, dtype=np.float32)
    return nonzero.mean(axis=0)


def bbox_diagonal(vector: np.ndarray) -> float:
    points = vector.reshape(-1, 3)
    nonzero = points[np.any(points != 0, axis=1)]
    if len(nonzero) == 0:
        return 1.0
    mins, maxs = nonzero.min(axis=0), nonzero.max(axis=0)
    return float(np.linalg.norm(maxs - mins))


def select_features(vector: np.ndarray) -> np.ndarray:
    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    o = 0
    out[o : o + POSE_COUNT * 3] = vector[POSE_OFFSET : POSE_OFFSET + POSE_COUNT * 3]
    o += POSE_COUNT * 3
    out[o : o + HAND_COUNT * 3] = vector[LEFT_HAND_OFFSET : LEFT_HAND_OFFSET + HAND_COUNT * 3]
    o += HAND_COUNT * 3
    out[o : o + HAND_COUNT * 3] = vector[RIGHT_HAND_OFFSET : RIGHT_HAND_OFFSET + HAND_COUNT * 3]
    o += HAND_COUNT * 3
    for idx in LIP_LANDMARK_INDICES:
        base = FACE_OFFSET + idx * 3
        out[o : o + 3] = vector[base : base + 3]
        o += 3
    return out


def load_split(split_dir: Path):
    """Returns {clip_key: [(frame_num, image_path, label), ...]} for one
    train/valid/test split folder, sorted by frame_num within each clip."""
    ann_path = split_dir / "_annotations.coco.json"
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"] if c["id"] != 0}
    image_id_to_label = {}
    for ann in coco["annotations"]:
        cat_name = categories.get(ann["category_id"])
        if cat_name is None:
            continue
        image_id_to_label[ann["image_id"]] = cat_name

    clips = defaultdict(list)
    skipped_unparsed = 0
    skipped_unlabeled = 0
    for img in coco["images"]:
        label = image_id_to_label.get(img["id"])
        if label is None:
            skipped_unlabeled += 1
            continue
        m = FILENAME_RE.match(img["file_name"])
        if not m:
            skipped_unparsed += 1
            continue
        clip_key = m.group("clip")
        frame_num = int(m.group("frame"))
        clips[clip_key].append((frame_num, str(split_dir / img["file_name"]), label))

    for clip_key in clips:
        clips[clip_key].sort(key=lambda t: t[0])

    if skipped_unparsed or skipped_unlabeled:
        print(
            f"  {split_dir.name}: skipped {skipped_unparsed} images with an "
            f"unparseable filename, {skipped_unlabeled} with no annotation"
        )
    return clips


def extract_clip_windows(frames: list, landmarker) -> tuple:
    """frames: [(frame_num, path, label), ...] sorted by frame_num, all
    belonging to one clip. Returns (windows, majority_label)."""
    import mediapipe as mp
    from collections import Counter

    labels_seen = Counter(label for _, _, label in frames)
    majority_label = labels_seen.most_common(1)[0][0]
    if len(labels_seen) > 1:
        print(f"    WARNING: clip has mixed labels {dict(labels_seen)}, using majority: {majority_label}")

    clip_features = []
    for i, (_, path, _label) in enumerate(frames):
        frame_bgr = cv2.imread(path)
        if frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(i * 1000 / TARGET_FPS)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        raw = np.zeros(TOTAL_FLOATS, dtype=np.float32)
        has_pose = write_landmarks(
            raw, POSE_OFFSET, POSE_COUNT, result.pose_landmarks[0] if result.pose_landmarks else None
        )
        write_landmarks(
            raw, LEFT_HAND_OFFSET, HAND_COUNT,
            result.left_hand_landmarks[0] if result.left_hand_landmarks else None,
        )
        write_landmarks(
            raw, RIGHT_HAND_OFFSET, HAND_COUNT,
            result.right_hand_landmarks[0] if result.right_hand_landmarks else None,
        )
        write_landmarks(
            raw, FACE_OFFSET, FACE_COUNT, result.face_landmarks[0] if result.face_landmarks else None
        )

        normalized = normalize(raw, has_pose)
        clip_features.append(select_features(normalized))

    windows = []
    if len(clip_features) == 0:
        return windows, majority_label

    if len(clip_features) < WINDOW_SIZE:
        # Short clip — edge-pad up to WINDOW_SIZE rather than drop it
        # entirely, same convention as pad_or_truncate() in train.py.
        pad_count = WINDOW_SIZE - len(clip_features)
        clip_features = clip_features + [clip_features[-1]] * pad_count

    for start in range(0, len(clip_features) - WINDOW_SIZE + 1, DEFAULT_STRIDE):
        window = np.concatenate(clip_features[start : start + WINDOW_SIZE])
        windows.append(window)

    return windows, majority_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Folder containing train/, valid/, test/ subfolders")
    parser.add_argument("--out", default="roboflow_dataset.json")
    parser.add_argument("--splits", default="train,valid,test", help="Comma-separated split folder names to process")
    args = parser.parse_args()

    root = Path(args.root)
    all_clips = {}
    for split_name in args.splits.split(","):
        split_dir = root / split_name.strip()
        if not (split_dir / "_annotations.coco.json").exists():
            print(f"Skipping {split_dir} — no _annotations.coco.json found")
            continue
        print(f"Loading {split_dir} ...")
        split_clips = load_split(split_dir)
        # Prefix clip keys by split to avoid collisions across splits.
        for k, v in split_clips.items():
            all_clips[f"{split_name}:{k}"] = v

    clip_lengths = [len(v) for v in all_clips.values()]
    print(
        f"Found {len(all_clips)} clips across all splits "
        f"(frames per clip: min {min(clip_lengths)}, max {max(clip_lengths)}, "
        f"mean {sum(clip_lengths) / len(clip_lengths):.1f})"
    )

    landmarker = make_landmarker()
    samples = []
    unmapped_labels = set()
    try:
        for i, (clip_key, frames) in enumerate(all_clips.items()):
            windows, raw_label = extract_clip_windows(frames, landmarker)
            label = ROBOFLOW_LABEL_MAP.get(raw_label)
            if label is None:
                unmapped_labels.add(raw_label)
                continue
            for w in windows:
                # clip_key already includes the split prefix (e.g.
                # "train:16-YW_MOV"), so it's globally unique across
                # splits — written out here so later per-clip processing
                # (motion filtering, clip-level train/val splits) can
                # group these windows back to their source clip. Without
                # this field, every Roboflow-derived sample gets treated
                # as clip-id-less "legacy" data by anything that groups
                # by clip_id, which silently excludes it from that kind
                # of processing entirely — this bit us once already.
                samples.append({"label": label, "frames": w.tolist(), "clip_id": clip_key})
            if (i + 1) % 100 == 0:
                print(f"  processed {i + 1}/{len(all_clips)} clips")
    finally:
        landmarker.close()

    if unmapped_labels:
        print(
            f"WARNING: {len(unmapped_labels)} raw label(s) had no entry in "
            f"ROBOFLOW_LABEL_MAP and were skipped entirely: {unmapped_labels}. "
            "Add them to the map at the top of this script and re-run if you "
            "want those included."
        )

    labels = sorted(set(s["label"] for s in samples))
    dataset = {
        "createdAt": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "featureDim": FEATURE_DIM,
        "windowSize": WINDOW_SIZE,
        "stride": DEFAULT_STRIDE,
        "labels": labels,
        "samples": samples,
    }
    with open(args.out, "w") as f:
        json.dump(dataset, f)
    print(f"Wrote {len(samples)} windows across {len(labels)} labels to {args.out}")


if __name__ == "__main__":
    main()
