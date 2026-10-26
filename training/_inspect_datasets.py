import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")

for name in ["combined_dataset.json", "video_dataset.json", "fsl105_dataset.json",
             "combined_dataset_merged.json"]:
    path = os.path.join("training", name)
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue
    with open(path) as f:
        data = json.load(f)
    labels = data.get("labels", [])
    samples = data.get("samples", [])
    print(f"{name}: {len(labels)} labels, {len(samples)} samples, "
          f"featureDim={data.get('featureDim')}, windowSize={data.get('windowSize')}")

# Compare label sets between deployed 303-label vocabulary and combined dataset
with open("src/inference/labels.ts", "r", encoding="utf-8") as f:
    ts = f.read()
import re
start = ts.index("[")
end = ts.rindex("]")
deployed = re.findall(r'"([^"]*)"', ts[start:end])
print(f"\nINFERENCE_LABELS (deployed): {len(deployed)}")

with open("training/combined_dataset.json") as f:
    cd = json.load(f)
cd_labels = set(cd.get("labels", []))
dep_set = set(deployed)
print(f"combined dataset labels not in INFERENCE_LABELS ({len(cd_labels - dep_set)}):")
print(sorted(cd_labels - dep_set))
print(f"\nINFERENCE_LABELS not in combined dataset ({len(dep_set - cd_labels)}):")
print(sorted(dep_set - cd_labels))

# label_map for merged model
with open("training/train_out_merged/label_map.json") as f:
    lm = json.load(f)
merged_labels = lm["labels"]
print(f"\ntrain_out_merged label_map: {len(merged_labels)}")
print(f"merged - deployed ({len(set(merged_labels) - dep_set)}): {sorted(set(merged_labels) - dep_set)}")
print(f"deployed - merged ({len(dep_set - set(merged_labels))}): {sorted(dep_set - set(merged_labels))}")
