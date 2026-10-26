import json, os

model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "public", "models", "fsl_model", "model.json")

with open(model_path) as f:
    m = json.load(f)

# Fix InputLayer: batch_shape -> batch_input_shape
changed = False
for group in m.get("modelTopology", {}).get("model_config", {}).get("config", {}).get("layers", []):
    c = group.get("config", {})
    if "batch_shape" in c and "batch_input_shape" not in c:
        c["batch_input_shape"] = c.pop("batch_shape")
        changed = True
        print(f"Fixed InputLayer: batch_shape -> batch_input_shape")
        print(f"  value: {c.get('batch_input_shape')}")

if changed:
    with open(model_path, "w") as f:
        json.dump(m, f)
    print("Saved fixed model.json")
else:
    print("No changes needed")