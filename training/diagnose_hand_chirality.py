"""
PHASE 5 — Defect 2 chirality probe (controlled).

Defect 2 claims MediaPipe's handedness classifier assumes a MIRRORED (selfie)
input while useHolisticPipeline.ts feeds an UN-MIRRORED camera frame
(`ctx.drawImage(video, 0, 0, w, h)`, no transform). If true, the
`leftHandLandmarks` slot actually holds the subject's RIGHT hand.

The first attempt to measure this from the extracted .bin files was
INCONCLUSIVE: the Phase 4 fixtures are largely archival GROUP photographs, so
MediaPipe's pose tracks one person while its hand landmarks frequently belong
to somebody else in the frame (measured hand-to-wrist distances of 3.0-3.5
shoulder-widths). Cross-person comparisons carry no chirality information.

This probe fixes that by:
  1. reading MediaPipe's ACTUAL `handedness` category (not just the slot), and
  2. keeping only detections anatomically attributable to the SAME person as
     the pose -- a hand counts only if its wrist lies within MAX_WRIST_GAP
     shoulder-widths of one of that person's pose wrists, which is exactly the
     unit normalizeLandmarks scales by.

MediaPipe POSE landmarks are anatomical (15/16 are the subject's real wrists and
are not subject to the mirror assumption), so they give independent ground truth
for which side a hand is really on.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from extract_landmarks import (  # noqa: E402
    CANVAS_SIZE, HAND_COUNT, LEFT_HAND_OFFSET, LEFT_SHOULDER, POSE_COUNT,
    POSE_OFFSET, RIGHT_HAND_OFFSET, RIGHT_SHOULDER, TARGET_FPS,
    get_landmark_group, letterbox, make_landmarker,
)

POSE_L_WRIST = 15
POSE_R_WRIST = 16
MAX_WRIST_GAP = 0.75
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def write_group(out, offset, group, count=HAND_COUNT):
    if group is None:
        return False
    for i in range(min(count, len(group))):
        p = group[i]
        out[offset + i * 3 : offset + i * 3 + 3] = (p.x, p.y, p.z)
    return True


def normalize_like_app(vec):
    """Mirror of normalizeLandmarks.ts: shoulder-anchor + shoulder-width scale,
    applied PER POINT (the TS loop steps i += 3)."""
    pts = vec.reshape(-1, 3)
    ls = pts[LEFT_SHOULDER]
    rs = pts[RIGHT_SHOULDER]
    anchor = (ls + rs) / 2.0
    scale = float(np.linalg.norm(ls - rs))
    if scale <= 1e-4:
        scale = 1.0
    return ((pts - anchor) / scale).reshape(-1)


def hand_label(result, index):
    """MediaPipe's own handedness category for hand `index` (0 = first detected)."""
    for field in ("handedness", "handednesses"):
        cat = getattr(result, field, None)
        if cat:
            try:
                return cat[index][0].category_name, float(cat[index][0].score)
            except Exception:
                pass
    return None, None



def detect(landmarker, frame_bgr, ts):
    """Same wrapper frame_to_features() uses in extract_landmarks.py."""
    import mediapipe as mp

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return landmarker.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
    )


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "scripts/fixtures/poses/candidates")
    imgs = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not imgs:
        print(f"no images in {root}")
        return 1

    landmarker = make_landmarker()
    stats = {"Left": {"Left": 0, "Right": 0}, "Right": {"Left": 0, "Right": 0}}
    same_person = 0
    cross_person = 0
    rows = []

    try:
        for i, img in enumerate(imgs):
            frame = cv2.imread(str(img), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            frame = letterbox(frame, CANVAS_SIZE)
            res = detect(landmarker, frame, int(i * 1000 / TARGET_FPS))

            pose = get_landmark_group(res.pose_landmarks)
            if pose is None or len(pose) < POSE_COUNT:
                continue

            raw = np.zeros(225, dtype=np.float32)
            write_group(raw, POSE_OFFSET, pose, POSE_COUNT)
            hands = []
            for slot, field in (("Left", res.left_hand_landmarks), ("Right", res.right_hand_landmarks)):
                g = get_landmark_group(field)
                if g is None or len(g) < HAND_COUNT:
                    continue
                label, score = hand_label(res, len(hands))
                off = LEFT_HAND_OFFSET if slot == "Left" else RIGHT_HAND_OFFSET
                write_group(raw, off, g)
                hands.append((slot, off, label, score))
            if not hands:
                continue

            vec = normalize_like_app(raw)
            plw = vec[POSE_L_WRIST * 3 : POSE_L_WRIST * 3 + 3]
            prw = vec[POSE_R_WRIST * 3 : POSE_R_WRIST * 3 + 3]

            for slot, off, label, score in hands:
                wrist = vec[off : off + 3]
                d_l = float(np.linalg.norm(wrist - plw))
                d_r = float(np.linalg.norm(wrist - prw))
                gap = min(d_l, d_r)
                if gap > MAX_WRIST_GAP:
                    cross_person += 1
                    rows.append(
                        f"  {img.stem:<24} MP slot={slot:<5} label={str(label):<6} "
                        f"DISCARDED cross-person gap={gap:.2f}"
                    )
                    continue
                same_person += 1
                true_side = "Left" if d_l < d_r else "Right"
                if label in stats:
                    stats[label][true_side] += 1
                lab = f"{label}({score:.2f})" if label else "None"
                agree = "AGREE" if label == true_side else "MIRRORED"
                rows.append(
                    f"  {img.stem:<24} MP slot={slot:<5} label={lab:<11} "
                    f"true={true_side:<5} gap={gap:.3f}  {agree}"
                )
    finally:
        try:
            landmarker.close()
        except Exception:
            pass

    print("\n--- hand detections ---")
    for r in rows:
        print(r)

    print("\n" + "=" * 84)
    print(f"same-person hands used : {same_person}")
    print(f"cross-person discarded : {cross_person}  (gap > {MAX_WRIST_GAP} shoulder-widths)")
    print(f"MP label 'Left'  -> truly Left {stats['Left']['Left']}, truly Right {stats['Left']['Right']}")
    print(f"MP label 'Right' -> truly Left {stats['Right']['Left']}, truly Right {stats['Right']['Right']}")

    n_l = sum(stats["Left"].values())
    n_r = sum(stats["Right"].values())
    total = n_l + n_r
    if total == 0:
        print("\nINCONCLUSIVE: no same-person hand detections in this fixture set")
        return 1

    disagree = stats["Left"]["Right"] + stats["Right"]["Left"]
    agree = stats["Left"]["Left"] + stats["Right"]["Right"]
    print()
    if disagree > agree:
        print(f"VERDICT: MIRRORED — {disagree}/{total} of MediaPipe's hand labels are the")
        print("         opposite of the subject's true anatomical side. Defect 2 CONFIRMED:")
        print("         the classifier assumes a flipped input; the app feeds an unflipped one,")
        print("         so the hand slots must be swapped at the canonical mapping stage.")
        return 2
    print(f"VERDICT: CORRECT — {agree}/{total} labels match anatomy. No swap needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
