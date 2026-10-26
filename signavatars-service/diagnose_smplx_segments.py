"""
Diagnose the SMPL-X forearm/upper-arm segment-length defect.

For each sampled label:
  1. Read the RAW MediaPipe source (public/avatar-data/<label>.json) and
     measure shoulder->elbow and elbow->wrist in the SAME space the fitter
     uses (normalized: shoulder midpoint at origin, shoulder width = 1).
  2. Read the GENERATED SMPL-X output (public/avatar-data-smplx/<label>.smplx.json)
     and measure the same two segments from smplx_joints.
  3. Compare. If the raw ratio is already bad, the defect is in the SOURCE
     DATA. If the raw ratio is fine but the output ratio is bad, the defect
     is in the FITTER.

Also dumps the IK-relevant landmarks for one frame so the numbers can be
eyeballed.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "public", "avatar-data")
SMPLX_DIR = os.path.join(ROOT, "public", "avatar-data-smplx")

POSE_COUNT = 33
LEFT_HAND_OFFSET = POSE_COUNT * 3
HAND_COUNT = 21
RIGHT_HAND_OFFSET = (POSE_COUNT + HAND_COUNT) * 3

# MediaPipe pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16


def load_raw(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    frames = data.get("frames")
    if isinstance(frames, list) and len(frames) > 0 and isinstance(frames[0], (int, float)):
        # Flat concatenated buffer: [f0v0, f0v1, ..., f1v0, ...]
        return frames
    # Otherwise fall back to a list-of-frames.
    return frames


def frame_buffer(frame) -> np.ndarray:
    """Return a 345-float buffer for a raw frame, whatever shape it has."""
    if isinstance(frame, list):
        if len(frame) == 345:
            return np.asarray(frame, dtype=np.float64)
        if len(frame) == 115:
            return np.asarray(frame, dtype=np.float64).flatten()
        # list of [x, y, z] triples
        flat = []
        for p in frame:
            flat.extend(p[:3])
        return np.asarray(flat, dtype=np.float64)
    if isinstance(frame, dict):
        for key in ("buffer", "landmarks", "data", "values"):
            if key in frame:
                return frame_buffer(frame[key])
    raise ValueError(f"unrecognised frame shape: {type(frame)}")


def all_frame_buffers(raw) -> list[np.ndarray]:
    """Split a flat concatenated buffer into per-frame 345-float buffers."""
    flat = np.asarray(raw, dtype=np.float64)
    if flat.size % 345 != 0:
        raise ValueError(f"flat buffer size {flat.size} not divisible by 345")
    return flat.reshape(-1, 345)


def normalize(pts: np.ndarray) -> np.ndarray:
    """Same normalisation the app uses: shoulder midpoint at origin, width=1."""
    mid = (pts[L_SHOULDER] + pts[R_SHOULDER]) / 2.0
    width = np.linalg.norm(pts[L_SHOULDER] - pts[R_SHOULDER])
    if width < 1e-6:
        return pts - mid
    return (pts - mid) / width


def raw_segments(buf: np.ndarray):
    pts = buf[: (POSE_COUNT * 3)].reshape(POSE_COUNT, 3)
    pts = normalize(pts)
    out = {}
    for side, (sh, el, wr) in {
        "left": (L_SHOULDER, L_ELBOW, L_WRIST),
        "right": (R_SHOULDER, R_ELBOW, R_WRIST),
    }.items():
        upper = np.linalg.norm(pts[el] - pts[sh])
        fore = np.linalg.norm(pts[wr] - pts[el])
        out[side] = (upper, fore, fore / upper if upper > 1e-9 else float("nan"))
    return out


def smplx_segments(joints: np.ndarray, ik_layout: bool):
    """IK layout: 11/12 clav, 14/15 elbow, 16/17 wrist. Official: 13/14, 18/19, 20/21."""
    if ik_layout:
        idx = {"left": (11, 14, 16), "right": (12, 15, 17)}
    else:
        idx = {"left": (13, 18, 20), "right": (14, 19, 21)}
    out = {}
    for side, (sh, el, wr) in idx.items():
        g = lambda i: joints[i]  # noqa: E731
        upper = np.linalg.norm(g(el) - g(sh))
        fore = np.linalg.norm(g(wr) - g(el))
        out[side] = (upper, fore, fore / upper if upper > 1e-9 else float("nan"))
    return out


def main():
    labels = sys.argv[1:] or ["A", "ADOPT", "AGAIN"]
    print(f"{'label':<12} {'side':<6} {'RAW upper':>10} {'RAW fore':>10} {'RAW ratio':>10} "
          f"{'OUT upper':>10} {'OUT fore':>10} {'OUT ratio':>10}")
    print("-" * 92)
    for label in labels:
        raw_path = os.path.join(RAW_DIR, f"{label}.json")
        smplx_path = os.path.join(SMPLX_DIR, f"{label}.smplx.json")
        if not (os.path.exists(raw_path) and os.path.exists(smplx_path)):
            print(f"{label}: missing file")
            continue
        raw_frames = load_raw(raw_path)
        raw_bufs = all_frame_buffers(raw_frames)
        with open(smplx_path, "r", encoding="utf-8") as fh:
            smplx = json.load(fh)
        sframes = smplx["frames"]

        n = min(3, len(raw_bufs), len(sframes))
        for f in range(n):
            buf = raw_bufs[f]
            r = raw_segments(buf)
            J = np.asarray(sframes[f]["smplx_joints"], dtype=np.float64).reshape(-1, 3)
            tail = np.abs(J[24:]).sum()
            ik_layout = tail < 1e-6
            s = smplx_segments(J, ik_layout)
            for side in ("left", "right"):
                print(f"{label:<12} {side:<6} {r[side][0]:>10.4f} {r[side][1]:>10.4f} "
                      f"{r[side][2]:>10.3f} {s[side][0]:>10.4f} {s[side][1]:>10.4f} {s[side][2]:>10.3f}")
        print(f"  (layout: {'IK' if ik_layout else 'OFFICIAL'})")
        print()

    # Detailed dump of one frame for eyeballing.
    label = labels[0]
    raw_frames = load_raw(os.path.join(RAW_DIR, f"{label}.json"))
    raw_bufs = all_frame_buffers(raw_frames)
    buf = raw_bufs[0]
    pts = normalize(buf[: POSE_COUNT * 3].reshape(POSE_COUNT, 3))
    print(f"--- {label} frame 0 raw normalised pose landmarks (shoulder width = 1) ---")
    for name, i in [("L_shoulder", L_SHOULDER), ("R_shoulder", R_SHOULDER),
                    ("L_elbow", L_ELBOW), ("R_elbow", R_ELBOW),
                    ("L_wrist", L_WRIST), ("R_wrist", R_WRIST)]:
        print(f"  {name:<12} idx {i:>2}  {np.round(pts[i], 4)}")


if __name__ == "__main__":
    main()
