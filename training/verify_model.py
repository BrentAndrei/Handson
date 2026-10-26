"""Verify the deployed TF.js model structure."""
import json

with open("C:/Users/BRENT/Downloads/handson_fixed/handson/public/models/fsl_model/model.json") as f:
    manifest = json.load(f)

print("format:", manifest.get("format"))
print("generatedBy:", manifest.get("generatedBy"))

cfg = manifest["modelTopology"]["model_config"]["config"]
layers = cfg["layers"]
print("layers:", len(layers))
for i, l in enumerate(layers):
    c = l.get("config", {})
    print(f"  [{i}] {l.get('class_name')} name={c.get('name')} units={c.get('units')} activation={c.get('activation')}")

print("output_layers:", cfg.get("output_layers"))
wm = manifest.get("weightsManifest", [])
total = sum(w.get("size", 0) for g in wm for w in g["weights"])
print("weight groups:", len(wm))
print("total weight bytes:", total)

with open("C:/Users/BRENT/Downloads/handson_fixed/handson/src/inference/labels.ts") as f:
    content = f.read()
print("INFERENCE_LABELS count:", content.count('  "'))