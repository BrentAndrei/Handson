"""
Runs MediaPipe's Holistic Landmarker over pre-recorded video clips (e.g.
the FSL-105 dataset, or any folder of labeled sign videos) and produces
the SAME fsl_dataset.json schema the browser app's "Collect data" mode
exports. Run train.py against the output directly, or merge it with a
browser-recorded dataset first (see merge_datasets() below) to combine
your own recordings with a larger published dataset.

This deliberately mirrors, step for step, the browser pipeline:
  useHolisticPipeline.ts  -> HolisticLandmarker detect + pack
  mediapipe/pack.ts       -> zero-fill missing landmark groups
  normalizeLandmarks.ts   -> shoulder-anchor translate + scale
  features/featureSelect.ts -> reduce to pose+hands+lips (345 dims)
  features/windowBuffer.ts  -> slice into 16-frame windows, stride 6
If any of those five files change, mirror the change here too, or the
video-derived features and live-webcam features will silently drift
apart and the model will perform worse on one source than the other.

Setup:
    pip install -r requirements-extract.txt

Two input modes:
  --mode folder   Videos organized as <root>/<label>/*.mp4 (simplest —
                  use this for your own downloaded clips organized by
                  sign name).
  --mode csv      A CSV with a path column and a label column (for
                  FSL-105's train.csv/test.csv + labels.csv layout).
                  Check the actual FSL-105 CSV headers after downloading
                  it — --path-col/--label-col/--labels-csv let you match
                  whatever they turn out to be without editing the script.

Examples:
    python extract_landmarks.py --mode folder --root ./my_clips --out video_dataset.json

    python extract_landmarks.py --mode csv --root ./fsl105 \
        --csv ./fsl105/train.csv --path-col path --label-col label_id \
        --labels-csv ./fsl105/labels.csv --labels-id-col id --labels-name-col label \
        --out fsl105_dataset.json
"""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# --- Mirrors mediapipe/types.ts -------------------------------------------
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

# Same effective capture rate the live app runs at (see TARGET_FPS in
# useHolisticPipeline.ts). Videos are resampled to this so a sign performed
# at, say, 30fps in a source clip produces roughly the same number of
# frames-per-second-of-real-motion as a live 24fps webcam capture — without
# this, a clip's motion would look artificially fast or slow relative to
# what the model sees during live inference.
TARGET_FPS = 24

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
MODEL_CACHE_PATH = Path(__file__).parent / "holistic_landmarker.task"


def ensure_model() -> str:
    if not MODEL_CACHE_PATH.exists():
        print(f"Downloading holistic landmarker model to {MODEL_CACHE_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_CACHE_PATH)
    return str(MODEL_CACHE_PATH)


def make_landmarker():
    import mediapipe as mp
    from mediapipe.tasks.python import vision

    BaseOptions = mp.tasks.BaseOptions
    HolisticLandmarker = vision.HolisticLandmarker
    HolisticLandmarkerOptions = vision.HolisticLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = HolisticLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=ensure_model()),
        running_mode=VisionRunningMode.VIDEO,
        min_pose_detection_confidence=0.3,
        min_pose_landmarks_confidence=0.3,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.3,
    )
    return HolisticLandmarker.create_from_options(options)


def write_landmarks(target: np.ndarray, offset: int, count: int, landmarks) -> bool:
    """Mirrors pack.ts's writeLandmarks: zero-fills if absent, else copies
    x/y/z per point, returns whether the group was present."""
    if not landmarks:
        target[offset : offset + count * 3] = 0
        return False

    points = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks

    try:
        n = len(points)
    except TypeError:
        points = [points]
        n = len(points)

    if n == 0:
        target[offset : offset + count * 3] = 0
        return False

    for i in range(min(count, n)):
        p = points[i]
        base = offset + i * 3
        target[base] = getattr(p, 'x', 0.0)
        target[base + 1] = getattr(p, 'y', 0.0)
        target[base + 2] = getattr(p, 'z', 0.0)
    return True


def normalize(vector: np.ndarray, has_pose: bool) -> np.ndarray:
    """Mirrors normalizeLandmarks.ts: shoulder-anchor translate + shoulder-
    width scale, falling back to a bounding-box diagonal when pose is
    missing or the shoulder distance is degenerate."""
    out = vector.copy()

    def point(group_offset, idx):
        base = group_offset + idx * 3
        return vector[base : base + 3]

    if has_pose:
        ls, rs = point(POSE_OFFSET, LEFT_SHOULDER), point(POSE_OFFSET, RIGHT_SHOULDER)
        anchor = (ls + rs) / 2
        shoulder_dist = float(np.linalg.norm(ls - rs))
        if shoulder_dist > 1e-4:
            scale = shoulder_dist
        else:
            scale = bbox_diagonal(vector)
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
    """Mirrors features/featureSelect.ts's selectFeatures."""
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


def get_landmark_group(landmarks_field):
    """Returns the flat list of landmark points for one detected person,
    regardless of whether the installed mediapipe python binding returns
    it already-flat ([point0, point1, ...]) or double-nested
    ([[point0, point1, ...]], i.e. one list of points per detected
    person, matching the JS API's shape).

    BUGFIX: the previous version always assumed double-nesting and did
    `landmarks_field[0]` unconditionally. When the installed mediapipe
    python package actually returns an already-flat list, that indexing
    grabs a single landmark *point* (not the list of points), which then
    got treated as a length-1 list downstream -- silently zeroing out
    every landmark index except 0 for pose/hands/face in the whole
    dataset. This was confirmed by diagnose_dataset.py: 96/99 pose dims,
    61/63 dims per hand, and 117/120 lip dims were dead (permanently
    zero) across all 29,820 windows -- i.e. almost no real hand-shape
    signal ever reached the model.
    """
    if not landmarks_field:
        return None
    if isinstance(landmarks_field, (list, tuple)) and len(landmarks_field) > 0:
        first = landmarks_field[0]
        # If landmarks_field[0] is itself a real landmark point (has an
        # .x attribute), the field was already flat and landmarks_field
        # IS the list of points -- don't unwrap further.
        if hasattr(first, "x"):
            return landmarks_field
        # Otherwise landmarks_field[0] correctly unwrapped a per-person
        # outer list, and `first` is the real list of points.
        return first
    return landmarks_field


def frame_to_features(frame_bgr, landmarker, timestamp_ms: int):
    """Runs ONE frame through the exact browser pipeline and returns
    (feature345, has_pose, has_left_hand, has_right_hand, has_face).

    Single source of truth shared by the video and image modes, so the .bin
    fixtures Phase 4 validates are produced by identical math to the
    video-derived training data:
        detect -> write_landmarks (pack.ts) -> normalize (normalizeLandmarks.ts)
               -> select_features (featureSelect.ts)
    """
    import mediapipe as mp

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    raw = np.zeros(TOTAL_FLOATS, dtype=np.float32)

    has_pose = write_landmarks(
        raw, POSE_OFFSET, POSE_COUNT, get_landmark_group(result.pose_landmarks),
    )
    has_left = write_landmarks(
        raw, LEFT_HAND_OFFSET, HAND_COUNT, get_landmark_group(result.left_hand_landmarks),
    )
    has_right = write_landmarks(
        raw, RIGHT_HAND_OFFSET, HAND_COUNT, get_landmark_group(result.right_hand_landmarks),
    )
    has_face = write_landmarks(
        raw, FACE_OFFSET, FACE_COUNT, get_landmark_group(result.face_landmarks),
    )

    normalized = normalize(raw, has_pose)
    return select_features(normalized), has_pose, has_left, has_right, has_face


def extract_clip_features(video_path: str, landmarker) -> list:
    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or TARGET_FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

    duration_sec = frame_count / native_fps
    target_frame_count = max(1, round(duration_sec * TARGET_FPS))
    sample_indices = np.linspace(0, frame_count - 1, target_frame_count).astype(int)
    sample_indices = np.unique(sample_indices)

    clip_features = []
    for i, frame_idx in enumerate(sample_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        import mediapipe as mp

        timestamp_ms = int(i * 1000 / TARGET_FPS)
        features, _pose, _lh, _rh, _fc = frame_to_features(
            frame_bgr, landmarker, timestamp_ms
        )
        clip_features.append(features)

    cap.release()
    return clip_features


def slice_into_windows(clip: list, window_size: int = WINDOW_SIZE, stride: int = DEFAULT_STRIDE):
    windows = []
    if len(clip) < window_size:
        return windows
    for start in range(0, len(clip) - window_size + 1, stride):
        window = np.concatenate(clip[start : start + window_size])
        windows.append(window)
    return windows


def collect_clip_paths_folder(root: str):
    pairs = []
    root_path = Path(root)
    for label_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        for clip_path in sorted(label_dir.glob("*")):
            if clip_path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                pairs.append((str(clip_path), label_dir.name))
    return pairs


def collect_clip_paths_csv(args):
    import pandas as pd

    df = pd.read_csv(args.csv)
    if args.labels_csv:
        labels_df = pd.read_csv(args.labels_csv)
        id_to_name = dict(zip(labels_df[args.labels_id_col], labels_df[args.labels_name_col]))
        df[args.label_col] = df[args.label_col].map(id_to_name)

    pairs = []
    for _, row in df.iterrows():
        rel_path = row[args.path_col]
        full_path = str(Path(args.root) / rel_path) if args.root else str(rel_path)
        pairs.append((full_path, str(row[args.label_col])))
    return pairs


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
# MediaPipe's HolisticLandmarker in VIDEO mode is STATEFUL: its
# SegmentationSmoothingCalculator RET_CHECKs that consecutive frames share the
# same height/width ("current_mat->rows == previous_mat->rows (640 vs. 722)").
# Photos all have different resolutions, so every image must be mapped onto one
# fixed canvas. We letterbox (scale to fit + pad) rather than squash, so body
# proportions survive -- normalizeLandmarks.ts then anchors at the shoulder
# midpoint and scales by shoulder width, which cancels the letterbox offset and
# scale anyway.
CANVAS_SIZE = 640


def letterbox(frame, size: int = CANVAS_SIZE):
    h, w = frame.shape[:2]
    scale = size / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas


def collect_image_paths(input_dir: str):
    root = Path(input_dir)
    if not root.is_dir():
        return []
    return sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def run_image_mode(input_dir: str, output_dir: str, require_pose: bool = True) -> int:
    """PHASE 4: turn pose photos into 345-float .bin landmark buffers.

    Each .bin is exactly FEATURE_DIM little-endian float32 values produced by
    the same pack -> normalize -> selectFeatures chain the browser runs, so
    scripts/verify-retarget.ts can feed them straight into
    mediapipeToCanonicalPose() (POSE_OFFSET=0 / LEFT_HAND_OFFSET=99 /
    RIGHT_HAND_OFFSET=162 are identical in both layouts).
    """
    paths = collect_image_paths(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"PHASE 4 image extraction: {len(paths)} image(s) in {input_dir}")
    print(f"featureDim={FEATURE_DIM} floats ({FEATURE_DIM * 4} bytes per .bin)\n")

    landmarker = make_landmarker()
    entries = []
    written = 0
    skipped = 0
    try:
        for i, img_path in enumerate(paths):
            frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"  [{i+1}/{len(paths)}] UNREADABLE, skipping: {img_path.name}")
                skipped += 1
                continue

            h, w = frame.shape[:2]
            # fixed canvas -- see CANVAS_SIZE note above (stateful VIDEO graph)
            frame = letterbox(frame)
            src_w, src_h = w, h

            # monotonically increasing timestamps, as VIDEO mode requires
            ts = int(i * 1000 / TARGET_FPS)
            try:
                feats, has_pose, has_l, has_r, has_f = frame_to_features(
                    frame, landmarker, ts
                )
            except Exception as e:  # one bad photo must not kill the batch
                print(f"  [{i+1}/{len(paths)}] {img_path.name}: DETECTION FAILED — {e}")
                skipped += 1
                continue

            pose_slice = feats[: POSE_COUNT * 3]
            live = int(np.count_nonzero(np.abs(pose_slice) > 1e-6))
            entry = {
                "image": img_path.name,
                "sourcePixels": [int(src_w), int(src_h)],
                "canvasPixels": [CANVAS_SIZE, CANVAS_SIZE],
                "hasPose": bool(has_pose),
                "hasLeftHand": bool(has_l),
                "hasRightHand": bool(has_r),
                "hasFace": bool(has_f),
                "poseDimsNonZero": live,
                "poseDimsTotal": POSE_COUNT * 3,
                "finite": bool(np.all(np.isfinite(feats))),
            }

            if require_pose and not has_pose:
                print(
                    f"  [{i+1}/{len(paths)}] {img_path.name}: NO POSE DETECTED "
                    f"(live pose dims {live}/{POSE_COUNT * 3}) -> no .bin written"
                )
                entry["bin"] = None
                entries.append(entry)
                skipped += 1
                continue

            bin_path = out / f"{img_path.stem}.bin"
            feats.astype(np.float32).tofile(bin_path)
            entry["bin"] = bin_path.name
            entries.append(entry)
            written += 1
            print(
                f"  [{i+1}/{len(paths)}] {img_path.name} -> {bin_path.name} "
                f"({FEATURE_DIM} floats, pose {live}/{POSE_COUNT * 3} live, "
                f"hands L={int(has_l)} R={int(has_r)}, face={int(has_f)})"
            )
    finally:
        try:
            landmarker.close()
        except Exception:
            pass  # a poisoned graph must not mask the real results

    with open(out / "manifest.json", "w") as f:
        json.dump(
            {
                "featureDim": FEATURE_DIM,
                "normalization": "shoulder-anchor translate + shoulder-width scale (normalizeLandmarks.ts)",
                "entries": entries,
            },
            f,
            indent=2,
        )

    nonfinite = [e["image"] for e in entries if not e["finite"]]
    print(f"\n============ PHASE 4 LANDMARK EXTRACTION SUMMARY ============")
    print(f"images scanned        : {len(paths)}  (canvas {CANVAS_SIZE}x{CANVAS_SIZE})")
    print(f".bin files written    : {written}  -> {out}")
    print(f"skipped (no pose/etc) : {skipped}")
    print(f"non-finite buffers    : {len(nonfinite)}  {nonfinite if nonfinite else ''}")
    print(f"bytes per .bin        : {FEATURE_DIM} x 4 = {FEATURE_DIM * 4}")
    print(f"manifest              : {out / 'manifest.json'}")
    if written == 0:
        print("FAIL: no landmark buffers produced")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["folder", "csv", "images"])
    parser.add_argument("--root", help="Video root directory")
    parser.add_argument("--out", default="video_dataset.json")
    parser.add_argument("--csv", help="CSV listing clip paths + labels (csv mode)")
    parser.add_argument("--path-col", default="path")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--labels-csv", help="Optional id->name label lookup CSV")
    parser.add_argument("--labels-id-col", default="id")
    parser.add_argument("--labels-name-col", default="label")
    parser.add_argument("--input_dir", help="Directory of pose photos (images mode)")
    parser.add_argument("--output_dir", help="Directory for .bin buffers (images mode)")
    args = parser.parse_args()

    # --- PHASE 4: still-image landmark extraction --------------------------
    if args.mode == "images" or (args.input_dir and not args.mode):
        input_dir = args.input_dir
        output_dir = args.output_dir or "scripts/fixtures/poses/extracted"
        if not input_dir:
            parser.error("--input_dir is required for --mode images")
        raise SystemExit(run_image_mode(input_dir, output_dir))

    if not args.mode:
        parser.error("--mode is required (folder | csv | images)")
    if not args.root:
        parser.error("--root is required")

    if args.mode == "folder":
        pairs = collect_clip_paths_folder(args.root)
    else:
        if not args.csv:
            parser.error("--csv is required for --mode csv")
        pairs = collect_clip_paths_csv(args)

    print(f"Found {len(pairs)} clips across {len(set(l for _, l in pairs))} labels")

    samples = []
    for i, (path, label) in enumerate(pairs):
        if not os.path.exists(path):
            print(f"  [{i+1}/{len(pairs)}] MISSING, skipping: {path}")
            continue

        landmarker = make_landmarker()
        try:
            clip_features = extract_clip_features(path, landmarker)
        finally:
            landmarker.close()

        windows = slice_into_windows(clip_features)
        # Keep every overlapping window from a recording tied to that
        # recording. train.py uses this to ensure a take cannot be present in
        # both its train and validation partitions.
        clip_id = f"clip_{i:05d}"
        for w in windows:
            samples.append({"label": label, "clip_id": clip_id, "frames": w.tolist()})
        print(f"  [{i+1}/{len(pairs)}] {path} -> {len(windows)} windows ({label})")

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


def merge_datasets(*paths: str, out: str) -> None:
    merged = None
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        if merged is None:
            merged = data
            continue
        if (data["featureDim"], data["windowSize"], data["stride"]) != (
            merged["featureDim"], merged["windowSize"], merged["stride"]
        ):
            raise ValueError(
                f"{path} has different featureDim/windowSize/stride than the "
                "first dataset — regenerate one of them so they match before merging."
            )
        merged["samples"].extend(data["samples"])
    merged["labels"] = sorted(set(s["label"] for s in merged["samples"]))
    with open(out, "w") as f:
        json.dump(merged, f)
    print(f"Merged {len(paths)} datasets -> {out} ({len(merged['samples'])} windows)")


if __name__ == "__main__":
    main()
