"""Test the ORIGINAL tfjs_model_399 model.json + .bin in Python TF
to see whether it recognizes HELLO. If the original model works and
the normalized one doesn't, the normalization script broke it.

Loads the tfjs layers model directly with tfjs (Python) — no need to
rebuild the architecture.
"""
import json
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# Load the tfjs model manifest + weights manually via tfjs_loader
from tensorflowjs import load_weights

manifest_path = "C:/Users/BRENT/Downloads/handson_fixed/handson/training/model_out/tfjs_model_399/model.json"
bin_path = "C:/Users/BRENT/Downloads/handson_fixed/handson/training/model_out/tfjs_model_399/group1-shard1of1.bin"

with open(manifest_path) as f:
    manifest = json.load(f)

# Build the model from the topology
from tensorflow.keras.models import model_from_json
topology = json.dumps(manifest["modelTopology"])
model = model_from_json(topology)
print("Model built OK, layers:", len(model.layers))

# Load weights
with open(bin_path, "rb") as f:
    weights = np.frombuffer(f.read(), dtype=np.float32)
print("weight bytes:", weights.size)

# Assign weights in order
flat_weights = []
idx = 0
for w in manifest["weightsManifest"][0]["weights"]:
    shape = w["shape"]
    size = int(np.prod(shape))
    flat_weights.append(tf.Variable(weights[idx:idx+size].reshape(shape), name=w["name"].replace("/", "_")))
    idx += size
print("assigned", len(flat_weights), "weight tensors")

# Load a HELLO training sample
sample_path = "C:/Users/BRENT/AppData/Local/Temp/kilo/training-sample-HELLO.json"
with open(sample_path) as f:
    sample = json.load(f)
X = np.asarray(sample["frames"], dtype=np.float32).reshape(1, 16, 345)

# Load labels
with open("C:/Users/BRENT/Downloads/handson_fixed/handson/src/inference/labels.ts") as f:
    content = f.read()
import re
m = content.split("export const INFERENCE_LABELS: string[] = [", 1)[1].rsplit("];", 1)[0]
labels = re.findall(r'"((?:[^"\\]|\\.)*)"', m)
print("labels:", len(labels), "HELLO idx:", labels.index("HELLO") if "HELLO" in labels else -1)

probs = model.predict(X, verbose=0)[0]
ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
print("top-5:", [(labels[i], round(float(probs[i]), 4)) for i in ranked[:5]])
hel = labels.index("HELLO")
print(f"HELLO prob: {probs[hel]:.4f} rank: {ranked.index(hel)+1}")