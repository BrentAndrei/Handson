"""Regenerate src/inference/labels.ts from train_out_merged/label_map.json."""
import json
import os

BASE = "C:/Users/BRENT/Downloads/handson_fixed/handson"
label_map_path = os.path.join(BASE, "training", "train_out_merged", "label_map.json")
out_path = os.path.join(BASE, "src", "inference", "labels.ts")

with open(label_map_path) as f:
    d = json.load(f)
labels = d["labels"]
print(f"labels: {len(labels)}")

lines = []
lines.append("/**")
lines.append(" * The classifier's label vocabulary, in the EXACT order the model's")
lines.append(" * output layer was trained with.")
lines.append(" *")
lines.append(" * This list lived inline in App.tsx until the capture studio needed")
lines.append(" * it too (to test a captured clip against the classifier before")
lines.append(" * installing it). The order is a contract: index i of a prediction")
lines.append(" * vector is this list's i-th label, so it must never be sorted,")
lines.append(" * filtered, or deduplicated. `label_map.json` does not ship with")
lines.append(" * the exported model, so this array is the source of truth.")
lines.append(" *")
lines.append(" * Regenerated from train_out_merged/label_map.json after the")
lines.append(" * merged-dataset training run (399 labels, 51,657 windows).")
lines.append(" */")
lines.append("export const INFERENCE_LABELS: string[] = [")
for lbl in labels:
    esc = lbl.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'  "{esc}",')
lines.append("];")
lines.append("")

with open(out_path, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {len(lines)} lines to {out_path}")
print("First 5:", labels[:5])
print("Last 5:", labels[-5:])