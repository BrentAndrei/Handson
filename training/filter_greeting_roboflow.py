"""
filter_greeting_roboflow.py

Removes Roboflow-sourced samples for 3 greeting labels from an existing
filtered dataset, producing a new dataset file.
"""

import json
from collections import defaultdict
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "combined_dataset_filtered.json"
OUTPUT_PATH = Path(__file__).parent / "combined_dataset_filtered_v3.json"

EXCLUDE_LABELS = {"GOOD MORNING", "GOOD AFTERNOON", "GOOD EVENING"}
ROBOFLOW_PREFIXES = ("train:", "valid:", "test:")


def is_roboflow(clip_id):
    return any(clip_id.startswith(p) for p in ROBOFLOW_PREFIXES)


def main():
    print(f"Loading {DATASET_PATH} ...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    original = data["samples"]
    kept = []
    removed = 0
    for sample in original:
        clip_id = sample.get("clip_id", "")
        if sample["label"] in EXCLUDE_LABELS and is_roboflow(clip_id):
            removed += 1
        else:
            kept.append(sample)

    print(f"Removed {removed} Roboflow-sourced samples from greeting labels")
    print(f"Final sample count: {len(kept)}")

    kept_counts = defaultdict(int)
    for s in kept:
        kept_counts[s["label"]] += 1

    print("\nPer-label counts for greeting labels:")
    for lbl in sorted(EXCLUDE_LABELS):
        print(f"  {lbl}: {kept_counts[lbl]}")
    print("(These should match FSL-105-only counts)")
    print(f"\nTotal labels: {len(set(kept_counts.keys()))}")

    out = {
        "createdAt": data.get("createdAt", ""),
        "featureDim": data["featureDim"],
        "windowSize": data["windowSize"],
        "stride": data.get("stride"),
        "labels": sorted(set(kept_counts.keys())),
        "samples": kept,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
