import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

# Check model_out label_map (399 labels)
model_out_map = os.path.join(base, "training", "model_out", "label_map.json")
with open(model_out_map) as f:
    lm = json.load(f)
labels = lm["labels"]
print(f"model_out label_map: {len(labels)} labels")
for i, label in enumerate(labels):
    print(f"  {i}: {label}")

# Check deployed model's actual shape
deployed = os.path.join(base, "public", "models", "fsl_model", "model.json")
with open(deployed) as f:
    dm = json.load(f)
# Get output layer size
layers = dm.get("modelTopology", {}).get("config", {}).get("layers", [])
for layer in layers:
    config = layer.get("config", {})
    if "units" in config:
        print(f"Layer {layer.get('class_name', '?')} {layer.get('name', '?')}: units={config['units']}")