import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

with open(os.path.join(base, "training", "tfjs_model_old_backup", "model.json")) as f:
    old = json.load(f)
with open(os.path.join(base, "public", "models", "fsl_model", "model.json")) as f:
    new = json.load(f)

old_layers = old["modelTopology"]["model_config"]["config"]["layers"]
new_layers = new["modelTopology"]["model_config"]["config"]["layers"]

for i, (o, n) in enumerate(zip(old_layers, new_layers)):
    oi = o.get("inbound_nodes")
    ni = n.get("inbound_nodes")
    print(f"\n=== Layer {i}: {o.get('class_name')} / {n.get('class_name')} ===")
    print(f"OLD inbound_nodes:")
    print(json.dumps(oi, indent=2)[:800])
    print(f"\nNEW inbound_nodes:")
    print(json.dumps(ni, indent=2)[:800])