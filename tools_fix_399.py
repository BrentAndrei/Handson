import json, os, copy

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"
NEW_MODEL = os.path.join(base, "training", "model_out", "tfjs_model_399", "model.json")
OUT_JSON = os.path.join(base, "public", "models", "fsl_model", "model.json")

with open(NEW_MODEL) as f:
    m = json.load(f)

# Fix 1: rename weight names in manifest to match v4.22 format
# v3.18: bidirectional/forward_gru/gru_cell/kernel
# v4.22: bidirectional/forward_gru/kernel
rename_map = {
    "/gru_cell/": "/",
}
for group in m.get("weightsManifest", []):
    new_weights = []
    for w in group.get("weights", []):
        new_name = w["name"]
        for old, new in rename_map.items():
            new_name = new_name.replace(old, new)
        if new_name != w["name"]:
            w = dict(w)  # copy
            w["name"] = new_name
        new_weights.append(w)
    group["weights"] = new_weights

print("Renamed weights:", [(w["name"]) for w in m["weightsManifest"][0]["weights"][:6]])

# Fix 2: inbound_nodes format - v4.22 uses simple array format
for layer in m["modelTopology"]["model_config"]["config"]["layers"]:
    c = layer.get("config", {})
    nodes = layer.get("inbound_nodes")
    if nodes and isinstance(nodes, list) and len(nodes) > 0 and isinstance(nodes[0], dict):
        # Convert from dict format to array format
        new_nodes = []
        for node in nodes:
            args = node.get("args", [])
            if args and len(args) == 1 and "keras_history" in args[0].get("config", {}):
                kh = args[0]["config"]["keras_history"]
                kwargs = node.get("kwargs", {})
                # Format: [[layer_name, node_idx, 0, kwargs]]
                new_nodes.append([[kh[0], kh[1], kh[2], kwargs]])
            else:
                new_nodes.append(node)
        layer["inbound_nodes"] = new_nodes
        # Do NOT put inbound_nodes inside config (Keras 3 mistake)

# Fix 3: Remove backward_layer from Bidirectional configs
for layer in m["modelTopology"]["model_config"]["config"]["layers"]:
    c = layer.get("config", {})
    if layer.get("class_name") == "Bidirectional" and "backward_layer" in c:
        del c["backward_layer"]
        print(f"Removed backward_layer from {layer.get('name')}")

# Fix 4: batch_shape -> batch_input_shape for InputLayer
for layer in m["modelTopology"]["model_config"]["config"]["layers"]:
    c = layer.get("config", {})
    if layer.get("class_name") == "InputLayer" and "batch_shape" in c:
        c["batch_input_shape"] = c.pop("batch_shape")
        print("Fixed InputLayer batch_input_shape")

# Fix 5: dtype from DTypePolicy dict to string
for layer in m["modelTopology"]["model_config"]["config"]["layers"]:
    c = layer.get("config", {})
    if isinstance(c.get("dtype"), dict):
        dtype_info = c["dtype"]
        c["dtype"] = dtype_info.get("config", {}).get("name", "float32")

with open(OUT_JSON, "w") as f:
    json.dump(m, f)

print(f"Saved fixed model.json to {OUT_JSON}")

# Verify the fix
with open(OUT_JSON) as f:
    v = json.load(f)
layers = v["modelTopology"]["model_config"]["config"]["layers"]
for i, L in enumerate(layers):
    if i == 0:
        print(f"Layer 0: {L['config'].get('batch_input_shape')}")
    if L.get("class_name") == "Bidirectional":
        print(f"Layer {i}: Bidirectional config keys = {list(L.get('config', {}).keys())}")
        if 'backward_layer' in L.get('config', {}):
            print("  ERROR: backward_layer still present!")