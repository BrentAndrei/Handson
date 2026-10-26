import json

with open("models/fsl_model/model.json") as f:
    m = json.load(f)

cfg = m.get('modelTopology', {}).get('model_config', {}).get('config', {})
layers = cfg.get('layers', [])
print(f"Number of layers: {len(layers)}")
for i, L in enumerate(layers):
    cn = L.get("class_name")
    name = L.get("name")
    c = L.get("config", {})
    bi = c.get("batch_input_shape")
    print(f"Layer {i}: {cn} name={name} batch_input_shape={bi}")

# Check the InputLayer first
IL = layers[0]
IC = IL.get("config", {})
print(f"\nInputLayer config keys: {list(IC.keys())}")
if "batch_input_shape" in IC:
    print(f"batch_input_shape: {IC['batch_input_shape']}")