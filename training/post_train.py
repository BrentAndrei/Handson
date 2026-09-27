import json
import os
import subprocess
import sys

OUT_DIR = "model_guiron1"
PUBLIC_DIR = r"C:\Users\BRENT\Downloads\handson_fixed\handson\public\models\fsl_model"

# 1. Read label_map.json
with open(os.path.join(OUT_DIR, "label_map.json")) as f:
    label_map = json.load(f)

labels = label_map["labels"]
feature_dim = label_map["featureDim"]
window_size = label_map["windowSize"]

print(f"Labels: {len(labels)}")
print(f"Feature dim: {feature_dim}, Window size: {window_size}")

# 2. Generate TypeScript INFERENCE_LABELS
ts_labels = "const INFERENCE_LABELS: string[] = [\n"
for label in labels:
    ts_labels += f'  "{label}",\n'
ts_labels += "];"

# Write to a temp file for easy copy-paste
with open("inference_labels.ts", "w") as f:
    f.write(ts_labels)
print("Wrote inference_labels.ts")

# 3. Convert H5 to TF.js
keras_model = os.path.join(OUT_DIR, "fsl_model.h5")
if not os.path.exists(keras_model):
    print(f"ERROR: {keras_model} not found")
    sys.exit(1)

tfjs_out = "model_guiron1_tfjs"
os.makedirs(tfjs_out, exist_ok=True)

cmd = [
    r"C:\Users\BRENT\Downloads\handson_fixed\handson\training\.venv-convert\Scripts\tensorflowjs_converter.exe",
    "--input_format=keras",
    "--output_format=tfjs_layers_model",
    "--quantize_float16=*",
    keras_model,
    tfjs_out,
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("ERROR:", result.stderr)
    sys.exit(1)

# 4. Copy to public/models/fsl_model/
import shutil
for f in os.listdir(tfjs_out):
    src = os.path.join(tfjs_out, f)
    dst = os.path.join(PUBLIC_DIR, f)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)

print(f"Model copied to {PUBLIC_DIR}")
print("Done!")
