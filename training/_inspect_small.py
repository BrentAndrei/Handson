import json, os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for name in ["fsl105_dataset.json"]:
    path = os.path.join("training", name)
    size = os.path.getsize(path)
    print(f"{name}: {size} bytes")

with open("training/train_out_merged/label_map.json") as f:
    lm = json.load(f)
labels = lm["labels"]
print(f"\ntrain_out_merged label_map: {len(labels)} labels")
print(f"first 20: {labels[:20]}")

# Check the model_out directory's label_map
with open("training/model_out/label_map.json") as f:
    lm2 = json.load(f)
labels2 = lm2["labels"]
print(f"\nmodel_out label_map: {len(labels2)} labels")
print(f"featureDim: {lm2.get('featureDim')}, windowSize: {lm2.get('windowSize')}")

# Diff label sets
set_merged = set(labels)
set_model_out = set(labels2)
print(f"\nmerged - model_out ({len(set_merged - set_model_out)}): {sorted(set_merged - set_model_out)}")
print(f"model_out - merged ({len(set_model_out - set_merged)}): {sorted(set_model_out - set_merged)}")
