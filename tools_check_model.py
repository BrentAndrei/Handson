import json

with open("models/fsl_model/model.json") as f:
    m = json.load(f)

cfg = m.get("modelTopology", {}).get("model_config", {}).get("config", {})
layers = cfg.get("layers", [])
print(f"Number of layers: {len(layers)}")
for i, L in enumerate(layers):
    cn = L.get("class_name")
    name = L.get("name")
    print(f"Layer {i}: {cn} name={name}")
    c = L.get("config", {})
    bi = c.get("batch_input_shape")
    ishape = c.get("input_shape")
    if bi:
        print(f"  batch_input_shape={bi}")
    if ishape:
        print(f"  input_shape={ishape}")
    if not bi and not ishape:
        print(f"  config keys: {list(c.keys())}")

# Check weightsManifest
wm = m.get("weightsManifest", [])
print(f"\nweightsManifest groups: {len(wm)}")
for g in wm:
    for w in g.get("weights", []):
        print(f"  weight: {w.get('name')} shape={w.get('shape')}")