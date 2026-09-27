"""
map_labels.py

Replaces numeric label IDs ("0".."104") in video_dataset.json with the real
FSL-105 sign names from labels.csv, producing a new dataset file with the
same schema train.py expects:

    { "featureDim": ..., "windowSize": ..., "labels": [...], "samples": [...] }

All other sample metadata, including ``clip_id`` used for leakage-free
training splits, is preserved unchanged.

Usage:
    python map_labels.py --dataset video_dataset.json --labels-csv labels.csv --out fsl_dataset.json
"""

import argparse
import csv
import json
import re


def clean_label(text: str) -> str:
    """Fix the mangled-apostrophe mojibake seen in the raw CSV
    (e.g. 'DONΓÇÖT UNDERSTAND' -> "DON'T UNDERSTAND"), and trim
    stray whitespace (e.g. 'NINE ' -> 'NINE')."""
    text = text.replace("ΓÇÖ", "'").replace("’", "'")
    return text.strip()


def load_id_to_label(csv_path: str) -> dict:
    id_to_label = {}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_label[row["id"].strip()] = clean_label(row["label"])
    return id_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="video_dataset.json")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--out", default="fsl_dataset.json")
    args = ap.parse_args()

    id_to_label = load_id_to_label(args.labels_csv)
    print(f"Loaded {len(id_to_label)} label mappings from {args.labels_csv}")

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    missing = set()
    for sample in data["samples"]:
        old_id = str(sample["label"]).strip()
        if old_id not in id_to_label:
            missing.add(old_id)
            continue
        sample["label"] = id_to_label[old_id]

    if missing:
        print(f"WARNING: {len(missing)} sample label id(s) had no match in labels.csv: {sorted(missing)}")

    # Rebuild the top-level labels list, ordered by numeric id so it stays
    # human-readable / stable (train.py itself recomputes labels via
    # sorted(set(...)) from the samples, so this field is for reference).
    ordered_ids = sorted(id_to_label.keys(), key=lambda x: int(x))
    data["labels"] = [id_to_label[i] for i in ordered_ids]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Wrote remapped dataset to {args.out}")
    print(f"Sample check -> first sample label: {data['samples'][0]['label']}")


if __name__ == "__main__":
    main()
