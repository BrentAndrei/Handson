"""
Prioritized capture plan — the "more data" half of the accuracy fix.

The model's weak spots are known (per_class_metrics_v4.json) but the
training dataset itself lives outside this repo, so this tool plans
new capture from the evaluation metrics plus whatever dataset JSONs
you point it at. It ranks every sign by how much accuracy headroom
it has (1 - F1) and how scarce its data is, and emits the capture
queue in order: which signs to record, how many clips each needs,
and why.

Two input sources, both optional but at least one required:

  --metrics  per_class_metrics_v4.json  (eval_model.py output).
             Classes here have measured precision/recall/F1/support.
  --dataset  fsl_dataset.json (repeatable).  train.py schema;
             gives exact per-class sample counts. Without it, the
             total is ESTIMATED from test support divided by
             --test-fraction (default 0.2 — the val_split train.py
             uses), so treat those numbers as planning estimates.

The sign universe is the avatar-data manifest (404 labels): a label
that has recordings but no metrics entry has NEVER been evaluated,
so its performance is unknown — it lands in the "unevaluated" band
at the mean assumed headroom, above healthy classes and below
measured-weak ones.

Usage:
    python capture_plan.py --metrics per_class_metrics_v4.json \
        --manifest ../public/avatar-data/manifest.json \
        --target-samples 150 --out capture_queue.json

Output: a ranked table on stdout and a machine-readable queue
(--out) the capture session can work through top-down.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

# F1 below this is "weak" — the class is misrecognizing signs, not
# just lacking data. Calibrated against the measured spread: the
# best weak class (COULD) sits at 0.66, the median class well
# above 0.9.
WEAK_F1 = 0.8


def load_metrics(path: Path) -> dict:
    with open(path, "r") as f:
        raw = json.load(f)
    # The file carries a summary row ("accuracy": 0.88) alongside
    # the per-class dicts; keep only the per-class entries.
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


def load_manifest_labels(path: Path) -> list:
    with open(path, "r") as f:
        data = json.load(f)
    labels = data.get("labels") or []
    if not labels and data.get("entries"):
        labels = [e["label"] for e in data["entries"] if e.get("label")]
    return labels


def count_dataset_samples(paths: list) -> Counter:
    counts: Counter = Counter()
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
        for sample in data.get("samples", []):
            counts[sample["label"]] += 1
    return counts


def plan(metrics: dict, universe: list, dataset_counts: Counter,
         target_samples: int, test_fraction: float) -> list:
    """Build the ranked capture queue. `universe` is every known
    sign label; `metrics` may cover only a subset of it."""
    rows = []
    # The unevaluated band is placed at the MEAN measured headroom,
    # not a fixed value: an unknown class is assumed average until
    # measured, so it never outranks a class measured as weak.
    measured_f1 = [float(m["f1-score"]) for m in metrics.values()]
    mean_headroom = (
        sum(1.0 - f for f in measured_f1) / len(measured_f1)
        if measured_f1 else 0.5
    )
    unevaluated_base = 2.0 * mean_headroom
    for label in sorted(set(universe) | set(metrics)):
        m = metrics.get(label)
        measured = m is not None
        f1 = float(m["f1-score"]) if measured else None
        support = int(m["support"]) if measured else 0

        if label in dataset_counts:
            total = dataset_counts[label]
            total_source = "dataset"
        elif measured:
            # Planning estimate: test support is ~test_fraction of
            # the class's samples (train.py splits by clip).
            total = round(support / test_fraction)
            total_source = "estimated"
        else:
            total = 0
            total_source = "unknown"

        clips_needed = max(0, target_samples - total)
        scarcity = clips_needed / target_samples

        if measured:
            headroom = 1.0 - f1
            priority = 2.0 * headroom + 0.5 * scarcity
            if f1 < WEAK_F1 and clips_needed > 0:
                category = "weak-and-scarce"
            elif f1 < WEAK_F1:
                category = "weak"
            elif clips_needed > 0:
                category = "scarce"
            else:
                category = "healthy"
        else:
            # No measurement: assume the mean measured headroom
            # and let scarcity order the band.
            priority = unevaluated_base + 0.5 * scarcity
            category = "unevaluated"

        rows.append({
            "label": label,
            "category": category,
            "priority": round(priority, 3),
            "f1": round(f1, 3) if f1 is not None else None,
            "test_support": support,
            "estimated_samples": total,
            "sample_source": total_source,
            "clips_needed": clips_needed,
        })

    rows.sort(key=lambda r: (-r["priority"], r["clips_needed"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics", type=Path,
                    help="per_class_metrics_v4.json (eval_model.py output)")
    ap.add_argument("--manifest", type=Path,
                    help="avatar-data manifest.json — the known sign universe")
    ap.add_argument("--dataset", type=Path, action="append", default=[],
                    help="fsl_dataset.json (train.py schema); repeatable")
    ap.add_argument("--target-samples", type=int, default=150,
                    help="per-class sample count to capture up to (default 150)")
    ap.add_argument("--test-fraction", type=float, default=0.2,
                    help="assumed test split when no dataset is given (default 0.2)")
    ap.add_argument("--out", type=Path,
                    help="write the machine-readable queue here as JSON")
    ap.add_argument("--top", type=int, default=30,
                    help="how many of the ranked rows to print (default 30)")
    args = ap.parse_args()

    if not args.metrics and not args.dataset:
        ap.error("need at least one of --metrics or --dataset")
    if not args.manifest:
        ap.error("--manifest is required (the sign universe to plan against)")

    metrics = load_metrics(args.metrics) if args.metrics else {}
    universe = load_manifest_labels(args.manifest)
    dataset_counts = count_dataset_samples(args.dataset)

    rows = plan(metrics, universe, dataset_counts,
                args.target_samples, args.test_fraction)

    by_cat = Counter(r["category"] for r in rows)
    total_clips = sum(r["clips_needed"] for r in rows)

    print(f"Sign universe: {len(universe)} labels "
          f"({len(metrics)} with metrics, {len(universe) - len(metrics)} unevaluated)")
    print(f"Target: {args.target_samples} samples/class, "
          f"test fraction {args.test_fraction:g}")
    print("Categories: " + ", ".join(f"{k}={v}" for k, v in sorted(by_cat.items())))
    print(f"Total capture queue: {total_clips} clips\n")

    header = (f"{'#':>3}  {'label':<32} {'category':<16} {'F1':>5} "
              f"{'have':>5} {'need':>5}  priority")
    print(header)
    print("-" * len(header))
    for r in rows[:args.top]:
        f1 = f"{r['f1']:.2f}" if r["f1"] is not None else "  -"
        print(f"{r['rank']:>3}  {r['label'][:32]:<32} {r['category']:<16} "
              f"{f1:>5} {r['estimated_samples']:>5} {r['clips_needed']:>5}  "
              f"{r['priority']:.3f}")
    if len(rows) > args.top:
        print(f"... {len(rows) - args.top} more rows in the queue")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({
                "target_samples": args.target_samples,
                "test_fraction": args.test_fraction,
                "total_clips": total_clips,
                "categories": dict(by_cat),
                "queue": rows,
            }, f, indent=1)
        print(f"\nqueue -> {args.out}")


if __name__ == "__main__":
    main()
