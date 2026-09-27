import json
import os
import re
import sys

# Read label_map.json from the training output
out_dir = sys.argv[1] if len(sys.argv) > 1 else "model_guiron1"
label_map_path = os.path.join(out_dir, "label_map.json")

with open(label_map_path) as f:
    label_map = json.load(f)

labels = label_map["labels"]
feature_dim = label_map["featureDim"]
window_size = label_map["windowSize"]

print(f"Labels: {len(labels)}")
print(f"Feature dim: {feature_dim}")
print(f"Window size: {window_size}")

# Generate TypeScript array for App.tsx
ts_array = "const INFERENCE_LABELS: string[] = [\n"
for i, label in enumerate(labels):
    ts_array += f'  "{label}",\n'
ts_array += "];"

print("\n--- INFERENCE_LABELS ---")
print(ts_array)
