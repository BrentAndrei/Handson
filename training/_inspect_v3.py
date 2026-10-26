import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson\training"

# Check fsl105_dataset.json
fsl105 = os.path.join(base, "fsl105_dataset.json")
print(f"fsl105_dataset.json: exists={os.path.exists(fsl105)}")

# Check combined_dataset.json size
cd_path = os.path.join(base, "combined_dataset.json")
if os.path.exists(cd_path):
    print(f"combined_dataset.json: size={os.path.getsize(cd_path)} bytes")

# Check train_out_merged label_map
merged_map = os.path.join(base, "train_out_merged", "label_map.json")
if os.path.exists(merged_map):
    with open(merged_map) as f:
        lm = json.load(f)
    labels = lm["labels"]
    print(f"\ntrain_out_merged label_map: {len(labels)} labels")
    print(f"first 20: {labels[:20]}")

# Check model_out label_map
model_out_map = os.path.join(base, "model_out", "label_map.json")
if os.path.exists(model_out_map):
    with open(model_out_map) as f:
        lm2 = json.load(f)
    labels2 = lm2["labels"]
    print(f"\nmodel_out label_map: {len(labels2)} labels")
    print(f"featureDim: {lm2.get('featureDim')}, windowSize: {lm2.get('windowSize')}")
    # Diff label sets - only if both exist
    if 'labels' in locals():
        set_merged = set(labels)
        set_model_out = set(labels2)
        print(f"\nmerged - model_out ({len(set_merged - set_model_out)}): {sorted(set_merged - set_model_out)}")
        print(f"model_out - merged ({len(set_model_out - set_merged)}): {sorted(set_model_out - set_merged)}")