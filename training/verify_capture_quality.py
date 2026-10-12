"""
Capture-quality gate — the "cleaner data" half of the accuracy fix.

The avatar recordings were captured with the camera rig angled so
the signer's nose projected far in front of the shoulder line: the
retargeter's own PHASE 21 audit measured the nose sitting 0.937
torso-lengths forward of the shoulder line, where a real
front-facing capture measures ~0.10-0.15. That is a capture-rig
defect, not a pose, and no amount of retargeting prior can recover
genuine forward lean from it (the upright-spine prior in
computeBodyPose exists precisely because the depth signal is junk).

This gate measures the same quantity on any dataset and fails the
classes that carry the defect, so new capture sessions can be
rejected (or re-shot) before the bad frames reach train.py.

Two input modes:

  --dataset fsl_dataset.json   train.py schema: samples of
                               windowSize x featureDim floats,
                               shoulder-normalized (the shape
                               extract_landmarks.py emits).
  --avatar-data <dir>          raw /avatar-data recordings: flat
                               frameCount x 345 float arrays per
                               file, labels from the manifest.

Per frame it measures:
  nose-forward   (shoulderMid.z - nose.z) / torsoLength
                 positive = nose toward the camera. Healthy
                 0.10-0.15; the known-bad rig measures 0.937.
  both-hands     fraction of frames where neither hand group is
                 the constant blob a zero-filled (absent) hand
                 group becomes after normalization — 21 identical
                 points means MediaPipe never saw that hand.
  degenerate     frames whose torso length is ~0 (no usable pose
                 estimate); they are skipped, not scored.

Usage:
    python verify_capture_quality.py --dataset fsl_dataset.json
    python verify_capture_quality.py --avatar-data ../public/avatar-data \
        --manifest ../public/avatar-data/manifest.json

Exit code 0 = every class passes; 1 = at least one class fails.

NOTE: the SHIPPED corpus (/avatar-data) fails this gate by design
— it was captured with the angled rig and predates the gate.
Every class measuring ~0.8-1.1 nose-forward is the known defect
the retargeter's PHASE 21 audit measured at 0.937 on ADOPT. The
gate exists so the NEXT capture session is rejected (or re-shot
facing the camera) before its frames reach train.py.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# --- Mirrors mediapipe/types.ts and training/extract_landmarks.py --
POSE_COUNT = 33
HAND_COUNT = 21
LIP_COUNT = 40  # the lip subset the app actually packs
POSE_OFFSET = 0
LEFT_HAND_OFFSET = POSE_OFFSET + POSE_COUNT * 3   # 99
RIGHT_HAND_OFFSET = LEFT_HAND_OFFSET + HAND_COUNT * 3  # 162
# Packed frame: 33 pose + 21x2 hand + 40 lip landmarks, x3.
FRAME_FLOATS = (POSE_COUNT + HAND_COUNT * 2 + LIP_COUNT) * 3  # 345

NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

# The PHASE 21 comment's healthy range is 0.10-0.15 torso-lengths;
# the corrupt rig measured 0.937. 0.30 is 2x the healthy ceiling —
# enough margin for a slightly angled capture, tight enough to catch
# the rig defect and tracking blowouts.
DEFAULT_MAX_NOSE_FORWARD = 0.30
# Signs are frequently two-handed; below half the frames carrying
# both hands means the clip is mostly one-handed or tracking is
# dropping a hand.
DEFAULT_MIN_BOTH_HANDS = 0.5

EPS = 1e-4


def point(buf, offset, index):
    base = offset + index * 3
    return buf[base], buf[base + 1], buf[base + 2]


def group_is_constant_blob(buf, offset, count):
    """An absent hand group is zero-filled at pack time; after
    shoulder normalization every one of its points maps to the same
    coordinate. 21 identical points is therefore "MediaPipe never
    saw this hand" in BOTH input spaces.

    `offset` is the group's FLOAT offset (99 / 162), exactly as
    mediapipe/types.ts and extract_landmarks.py define it — not a
    landmark index.
    """
    for i in range(1, count * 3):
        if abs(buf[offset + i] - buf[offset]) > 1e-6:
            return False
    return True


def frame_quality(buf):
    """(nose_forward | None, both_hands, degenerate) for one frame.
    nose_forward is None when the pose estimate is degenerate."""
    ls = point(buf, POSE_OFFSET, LEFT_SHOULDER)
    rs = point(buf, POSE_OFFSET, RIGHT_SHOULDER)
    lh = point(buf, POSE_OFFSET, LEFT_HIP)
    rh = point(buf, POSE_OFFSET, RIGHT_HIP)
    nose = point(buf, POSE_OFFSET, NOSE)

    shoulder_mid = tuple((a + b) / 2 for a, b in zip(ls, rs))
    hip_mid = tuple((a + b) / 2 for a, b in zip(lh, rh))

    torso = sum((h - s) ** 2 for h, s in zip(hip_mid, shoulder_mid)) ** 0.5
    if torso < EPS:
        return None, False, True

    # Positive = nose in front of the shoulder line (toward camera).
    nose_forward = (shoulder_mid[2] - nose[2]) / torso
    both_hands = not (
        group_is_constant_blob(buf, LEFT_HAND_OFFSET, HAND_COUNT)
        or group_is_constant_blob(buf, RIGHT_HAND_OFFSET, HAND_COUNT)
    )
    return nose_forward, both_hands, False


class ClassStats:
    def __init__(self):
        self.frames = 0
        self.degenerate = 0
        self.nose_sum = 0.0
        self.nose_max = 0.0
        self.both_hands = 0

    def add(self, nose_forward, both_hands, degenerate):
        self.frames += 1
        if degenerate:
            self.degenerate += 1
            return
        self.nose_sum += nose_forward
        self.nose_max = max(self.nose_max, nose_forward)
        if both_hands:
            self.both_hands += 1

    @property
    def valid(self):
        return self.frames - self.degenerate

    @property
    def nose_mean(self):
        return self.nose_sum / self.valid if self.valid else 0.0

    @property
    def both_hands_fraction(self):
        return self.both_hands / self.valid if self.valid else 0.0


def iterate_dataset(path: Path):
    """Yield (label, frame) pairs from a train.py-schema dataset."""
    with open(path, "r") as f:
        data = json.load(f)
    window = int(data.get("windowSize", 16))
    dim = int(data.get("featureDim", 345))
    per_sample = window * dim
    for sample in data.get("samples", []):
        frames = sample["frames"]
        for i in range(window):
            yield sample["label"], frames[i * dim:(i + 1) * dim]


def iterate_avatar_data(root: Path, manifest: Path | None):
    """Yield (label, frame) pairs from raw avatar-data recordings."""
    file_to_label = {}
    if manifest and manifest.exists():
        with open(manifest, "r") as f:
            m = json.load(f)
        for entry in m.get("entries", []):
            if entry.get("label") and entry.get("file"):
                file_to_label[entry["file"]] = entry["label"]

    for path in sorted(root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        label = file_to_label.get(path.name)
        if label is None:
            # Fall back to reversing the sanitizer: non-alphanumerics
            # became underscores.
            label = path.stem.replace("_", " ")
        with open(path, "r") as f:
            data = json.load(f)
        frames = data.get("frames", [])
        # Tolerate both the real layout (one flat float array) and
        # the legacy TS type (array of per-frame arrays).
        if frames and isinstance(frames[0], list):
            for frame in frames:
                yield label, frame
        else:
            for i in range(0, len(frames) - FRAME_FLOATS + 1, FRAME_FLOATS):
                yield label, frames[i:i + FRAME_FLOATS]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", type=Path,
                     help="fsl_dataset.json (train.py schema)")
    src.add_argument("--avatar-data", type=Path,
                     help="directory of raw avatar-data recordings")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="avatar-data manifest.json (label lookup)")
    ap.add_argument("--max-nose-forward", type=float,
                    default=DEFAULT_MAX_NOSE_FORWARD,
                    help=f"max mean nose-forward ratio per class (default {DEFAULT_MAX_NOSE_FORWARD})")
    ap.add_argument("--min-both-hands", type=float,
                    default=DEFAULT_MIN_BOTH_HANDS,
                    help=f"min fraction of frames with both hands (default {DEFAULT_MIN_BOTH_HANDS})")
    args = ap.parse_args()

    if args.dataset:
        frames = iterate_dataset(args.dataset)
    else:
        frames = iterate_avatar_data(args.avatar_data, args.manifest)

    stats: dict[str, ClassStats] = defaultdict(ClassStats)
    for label, frame in frames:
        nf, both, degenerate = frame_quality(frame)
        stats[label].add(nf, both, degenerate)

    if not stats:
        print("No frames found — nothing to check.")
        return

    failures = 0
    header = (f"{'label':<32} {'frames':>6} {'degen':>5} "
              f"{'noseMean':>8} {'noseMax':>7} {'2hands':>6}  verdict")
    print(header)
    print("-" * len(header))
    for label in sorted(stats):
        s = stats[label]
        reasons = []
        if s.nose_mean > args.max_nose_forward:
            reasons.append(
                f"nose-forward {s.nose_mean:.2f} > {args.max_nose_forward} "
                f"(angled rig, re-shoot facing the camera)")
        if s.both_hands_fraction < args.min_both_hands:
            reasons.append(
                f"both-hands {s.both_hands_fraction:.2f} < {args.min_both_hands}")
        ok = not reasons
        if not ok:
            failures += 1
        verdict = "PASS" if ok else "FAIL: " + "; ".join(reasons)
        print(f"{label[:32]:<32} {s.frames:>6} {s.degenerate:>5} "
              f"{s.nose_mean:>8.3f} {s.nose_max:>7.3f} "
              f"{s.both_hands_fraction:>6.2f}  {verdict}")

    print(f"\n{len(stats)} classes checked, {failures} failed "
          f"(max nose-forward {args.max_nose_forward}, "
          f"min both-hands {args.min_both_hands})")

    global_mean = (
        sum(s.nose_sum for s in stats.values())
        / max(1, sum(s.valid for s in stats.values()))
    )
    print(f"Global mean nose-forward: {global_mean:.3f} torso-lengths "
          f"(healthy capture: 0.10-0.15)")
    if global_mean > args.max_nose_forward:
        print("  ^ a corpus-wide value this high is a CAPTURE-RIG defect,")
        print("    not per-class noise: the whole corpus was shot with the")
        print("    camera angled off-axis. Re-shoot facing the camera; no")
        print("    retargeting prior can recover real forward lean from it.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
