import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

with open(os.path.join(base, "training", "tfjs_model_old_backup", "model.json")) as f:
    old = json.load(f)
with open(os.path.join(base, "public", "models", "fsl_model", "model.json")) as f:
    new = json.load(f)

# Fix the new model.json to match old format
def fix_model(m):
    """Normalize Keras 3.15.1 TF.js export to the older format that TF.js v4.22 can load."""
    cfg = m["modelTopology"]["model_config"]["config"]
    for layer in cfg.get("layers", []):
        c = layer.get("config", {})
        # Fix dtype field (DTypePolicy -> string)
        if isinstance(c.get("dtype"), dict):
            c["dtype"] = c["dtype"].get("config", {}).get("name", "float32")
        # Fix inbound_nodes format (new format -> old format)
        nodes = layer.get("inbound_nodes")
        if nodes and isinstance(nodes[0], dict) and "args" in nodes[0]:
            new_nodes = []
            for node in nodes:
                args = node.get("args", [])
                if args and isinstance(args[0], dict) and "keras_history" in args[0].get("config", {}):
                    kh = args[0]["config"]["keras_history"]
                    kwargs = node.get("kwargs", {})
                    new_nodes.append([[kh[0], kh[1], kh[2], kwargs]])
                else:
                    new_nodes.append(node)
            layer["inbound_nodes"] = new_nodes
        # Remove backward_layer from Bidirectional configs
        if "backward_layer" in c:
            del c["backward_layer"]
    return m

new = fix_model(new)

with open(os.path.join(base, "public", "models", "fsl_model", "model.json"), "w") as f:
    json.dump(new, f)
print("Fixed and saved model.json")

# Verify
with open(os.path.join(base, "public", "models", "fsl_model", "model.json")) as f:
    v = json.load(f)
layers = v["modelTopology"]["model_config"]["config"]["layers"]
for i, L in enumerate(layers):
    c = L.get("config", {})
    nodes = L.get("inbound_nodes")
    if nodes:
        print(f"Layer {i}: {L.get('class_name')} inbound_nodes[0] type={type(nodes[0]).__name__} dtype={c.get('dtype')}")