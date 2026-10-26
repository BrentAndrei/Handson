"""
Merge the 78 avatar-data labels that exist as individual JSON files
in public/avatar-data/ but are missing from combined_dataset.json.

Each avatar-data file is a flat list of 345-float frames (SMPL-X converted).
We chunk them into 16-frame windows with stride=8 (same as the YouTube pipeline)
and append to combined_dataset.json.

Usage:
    python merge_avatar_labels.py
"""
import json
from pathlib import Path

BASE = Path("C:/Users/BRENT/Downloads/handson_fixed/handson")
AVATAR_DIR = BASE / "public" / "avatar-data"
COMBINED_PATH = BASE / "training" / "combined_dataset.json"
OUT_PATH = BASE / "training" / "combined_dataset_merged.json"

WINDOW_SIZE = 16
STRIDE = 8
FEATURE_DIM = 345

# Load existing dataset (streaming-friendly: read labels only first)
print(f"Loading existing dataset from {COMBINED_PATH} ...")
with open(COMBINED_PATH, "rb") as f:
    existing = json.load(f)
existing_labels = set(existing["labels"])
existing_samples = existing["samples"]

print(f"Existing: {len(existing_labels)} labels, {len(existing_samples)} samples")

# Find avatar-data files whose stem is NOT in existing_labels
missing = []
for f in sorted(AVATAR_DIR.glob("*.json")):
    if f.stem not in existing_labels:
        missing.append(f)

print(f"Missing avatar-data files: {len(missing)}")

new_samples = []
new_labels = set()
total_frames = 0
skipped_no_frames = 0
skipped_short = 0

for f in missing:
    with open(f) as fh:
        data = json.load(fh)
    label = data.get("label", f.stem)
    frames = data.get("frames")
    if frames is None:
        skipped_no_frames += 1
        continue
    n_frames = len(frames) // FEATURE_DIM
    total_frames += n_frames

    # Chunk into windows
    for start in range(0, n_frames - WINDOW_SIZE + 1, STRIDE):
        window = frames[start * FEATURE_DIM : (start + WINDOW_SIZE) * FEATURE_DIM]
        if len(window) != WINDOW_SIZE * FEATURE_DIM:
            continue
        new_samples.append({"label": label, "frames": window, "clip_id": f.stem})
        new_labels.add(label)

print(f"Skipped (no frames key): {skipped_no_frames}")
print(f"New labels: {len(new_labels)}, new samples: {len(new_samples)}, total frames: {total_frames}")

# Merge
all_labels = sorted(existing_labels | new_labels)
all_samples = existing_samples + new_samples

merged = {
    "createdAt": existing.get("createdAt", ""),
    "featureDim": FEATURE_DIM,
    "windowSize": WINDOW_SIZE,
    "stride": STRIDE,
    "labels": all_labels,
    "samples": all_samples,
}

with open(OUT_PATH, "w") as f:
    json.dump(merged, f)

print(f"\nMerged dataset written to {OUT_PATH}")
print(f"Total: {len(all_labels)} labels, {len(all_samples)} samples")