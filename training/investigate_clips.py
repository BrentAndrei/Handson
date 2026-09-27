"""
Diagnostic script to investigate specific FSL-105 clips for possible
mislabeling / duplicate / outlier issues.

No modifications to training data or models — purely read/inspect.
"""

import hashlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np

FSL105_ROOT = Path(r"C:\Users\BRENT\Downloads\FSL-105\FSL-105 A dataset for recognizing 105 Filipino sign language videos")
OUTPUT_DIR = Path(__file__).parent / "spot_check_thumbnails"

TARGET_CLIPS = {
    "GREEN": [r"clips\73\5.MOV"],
    "GRANDMOTHER": [r"clips\57\13.MOV"],
    "FOUR": [r"clips\23\10.MOV"],
}

COMPARISON_CLIPS = {
    "GREEN": [r"clips\73\2.MOV", r"clips\73\0.MOV"],
    "GRANDMOTHER": [r"clips\57\18.MOV", r"clips\57\15.MOV"],
    "FOUR": [r"clips\23\11.MOV", r"clips\23\6.MOV"],
}


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def video_info(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    size = path.stat().st_size
    return {
        "frame_count": frame_count,
        "fps": fps,
        "duration_sec": duration,
        "size_bytes": size,
    }


def extract_frames(path: Path, out_dir: Path, percents=(10, 25, 50, 75, 90)):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for pct in percents:
        idx = int(frame_count * pct / 100)
        idx = min(idx, frame_count - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = out_dir / f"{pct}.jpg"
        cv2.imwrite(str(out_path), frame)
    cap.release()


def main():
    print("=== TARGET CLIPS ===\n")
    target_hashes = {}
    target_info = {}
    for label, rels in TARGET_CLIPS.items():
        for rel in rels:
            p = FSL105_ROOT / rel
            h = md5_of(p)
            target_hashes.setdefault(label, []).append((rel, h))
            info = video_info(p)
            target_info.setdefault(label, []).append((rel, info))
            print(f"{label}: {rel}")
            print(f"  MD5:     {h}")
            print(f"  Size:    {info['size_bytes']} bytes")
            print(f"  Frames:  {info['frame_count']}")
            print(f"  FPS:     {info['fps']:.3f}")
            print(f"  Duration:{info['duration_sec']:.3f} sec")
            print()

    print("=== COMPARISON CLIPS (same labels) ===\n")
    comp_info = {}
    for label, rels in COMPARISON_CLIPS.items():
        for rel in rels:
            p = FSL105_ROOT / rel
            info = video_info(p)
            comp_info.setdefault(label, []).append((rel, info))
            print(f"{label}: {rel}")
            print(f"  Size:    {info['size_bytes']} bytes")
            print(f"  Frames:  {info['frame_count']}")
            print(f"  FPS:     {info['fps']:.3f}")
            print(f"  Duration:{info['duration_sec']:.3f} sec")
            print()

    print("=== EXTRACTING EXTRA THUMBNAILS ===\n")
    for label, rels in TARGET_CLIPS.items():
        for rel in rels:
            p = FSL105_ROOT / rel
            stem = Path(rel).stem
            out_dir = OUTPUT_DIR / f"{label}_extra"
            extract_frames(p, out_dir)
            print(f"{label}: {rel} -> {out_dir}/")

    # Hash comparison summary
    print("\n=== HASH COMPARISON ===")
    all_hashes = []
    for label, items in target_hashes.items():
        for rel, h in items:
            all_hashes.append((label, rel, h))
    for i in range(len(all_hashes)):
        for j in range(i+1, len(all_hashes)):
            same = all_hashes[i][2] == all_hashes[j][2]
            print(f"{all_hashes[i][0]} vs {all_hashes[j][0]}: {'SAME HASH' if same else 'different'}")


if __name__ == "__main__":
    main()
