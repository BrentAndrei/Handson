import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

with open(os.path.join(base, "training", "tfjs_model_old_backup", "model.json")) as f:
    old = json.load(f)
with open(os.path.join(base, "public", "models", "fsl_model", "model.json")) as f:
    new = json.load(f)

old_layers = old["modelTopology"]["model_config"]["config"]["layers"]
new_layers = new["modelTopology"]["model_config"]["config"]["layers"]

# Check Bidirectional layer's 'layer' config
for i in [1, 2]:
    o = old_layers[i]
    n = new_layers[i]
    print(f"\n=== Layer {i}: Bidirectional ===")
    print(f"OLD Bidirectional config keys: {list(o.get('config', {}).keys())}")
    print(f"NEW Bidirectional config keys: {list(n.get('config', {}).keys())}")
    
    ol = o.get("config", {}).get("layer", {})
    nl = n.get("config", {}).get("layer", {})
    print(f"\nOLD GRU layer class_name: {ol.get('class_name')}")
    print(f"NEW GRU layer class_name: {nl.get('class_name')}")
    print(f"\nOLD GRU config keys: {list(ol.get('config', {}).keys())}")
    print(f"NEW GRU config keys: {list(nl.get('config', {}).keys())}")
    
    # Check specific GRU config fields
    for k in ["units", "activation", "recurrent_activation", "use_bias", "reset_after", "dropout", "recurrent_dropout"]:
        ov = ol.get("config", {}).get(k)
        nv = nl.get("config", {}).get(k)
        if ov != nv:
            print(f"  GRU config '{k}': old={ov} new={nv}")