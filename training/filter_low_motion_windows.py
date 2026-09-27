"""
filter_low_motion_windows.py

Per-clip motion filter for combined_dataset.json.

For each clip_id group, computes motion energy per window and keeps only
windows at or above that clip's own median energy. Samples without a
clip_id are excluded entirely. Writes combined_dataset_filtered.json.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


DATASET_PATH = Path(__file__).parent.parent.parent / "combined_dataset.json"
OUTPUT_PATH = Path(__file__).parent / "combined_dataset_filtered.json"


def compute_motion_energy(frames_2d: np.ndarray) -> float:
    diffs = np.diff(frames_2d, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.sum(dists))


def main():
    print(f"Loading {DATASET_PATH} ...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    feature_dim = data["featureDim"]
    window_size = data["windowSize"]
    expected = window_size * feature_dim
    stride = data.get("stride")
    labels = sorted(set(s["label"] for s in data["samples"]))

    samples = data["samples"]
    total_original = len(samples)
    no_clip = [s for s in samples if not s.get("clip_id")]
    with_clip = [s for s in samples if s.get("clip_id")]

    print(f"Original sample count: {total_original}")
    print(f"  With clip_id:    {len(with_clip)}")
    print(f"  Without clip_id: {len(no_clip)}")

    # Group by clip_id
    clips = defaultdict(list)
    for s in with_clip:
        clips[s["clip_id"]].append(s)

    # Compute motion energy per sample and find per-clip median
    clip_medians = {}
    sample_energy = {}
    for clip_id, group in clips.items():
        energies = []
        for s in group:
            frames = np.asarray(s["frames"], dtype=np.float32)
            if frames.size != expected:
                energies.append(0.0)
                continue
            frames_2d = frames.reshape(window_size, feature_dim)
            e = compute_motion_energy(frames_2d)
            energies.append(e)
            sample_energy[id(s)] = e
        if energies:
            clip_medians[clip_id] = float(np.median(energies))
        else:
            clip_medians[clip_id] = 0.0

    # Filter: keep samples at or above their clip's median energy
    kept = []
    removed = 0
    for s in with_clip:
        e = sample_energy.get(id(s), 0.0)
        median = clip_medians.get(s["clip_id"], 0.0)
        if e >= median:
            kept.append(s)
        else:
            removed += 1

    print(f"Removed by motion filter: {removed}")
    print(f"Final sample count: {len(kept)}")

    # Per-label counts
    kept_counts = defaultdict(int)
    for s in kept:
        kept_counts[s["label"]] += 1

    all_labels = sorted(set(kept_counts.keys()))
    print("\nPer-label window counts after filtering:")
    print(f"{'Label':<30} {'Count':>10}")
    print("-" * 42)
    for lbl in all_labels:
        print(f"{lbl:<30} {kept_counts[lbl]:>10}")
    print("-" * 42)
    print(f"{'TOTAL':<30} {sum(kept_counts.values()):>10}")

    # Write output
    out = {
        "createdAt": data.get("createdAt", ""),
        "featureDim": feature_dim,
        "windowSize": window_size,
        "stride": stride,
        "labels": labels,
        "samples": kept,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
