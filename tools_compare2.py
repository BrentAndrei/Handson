import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

with open(os.path.join(base, "training", "tfjs_model_old_backup", "model.json")) as f:
    old = json.load(f)
with open(os.path.join(base, "public", "models", "fsl_model", "model.json")) as f:
    new = json.load(f)

print("OLD generatedBy:", old.get("generatedBy"), "| convertedBy:", old.get("convertedBy"))
print("NEW generatedBy:", new.get("generatedBy"), "| convertedBy:", new.get("convertedBy"))
print("OLD keras_version:", old.get("modelTopology", {}).get("keras_version"))
print("NEW keras_version:", new.get("modelTopology", {}).get("keras_version"))

old_layers = old["modelTopology"]["model_config"]["config"]["layers"]
new_layers = new["modelTopology"]["model_config"]["config"]["layers"]

for i, (o, n) in enumerate(zip(old_layers, new_layers)):
    print(f"\n--- Layer {i}: {o.get('class_name')} / {n.get('class_name')} ---")
    oi = o.get("inbound_nodes")
    ni = n.get("inbound_nodes")
    if oi != ni:
        print(f"  OLD inbound_nodes: {json.dumps(oi)[:300]}")
        print(f"  NEW inbound_nodes: {json.dumps(ni)[:300]}")
    else:
        print(f"  inbound_nodes identical")

# training_config compare
otc = old["modelTopology"].get("training_config")
ntc = new["modelTopology"].get("training_config")
print(f"\nOLD training_config type: {type(otc).__name__} len={len(otc) if otc else 0}")
print(f"NEW training_config type: {type(ntc).__name__} len={len(ntc) if ntc else 0}")

# Compare Bidirectional layer configs
for idx in (1, 2):
    o = old_layers[idx]["config"]
    n = new_layers[idx]["config"]
    ok = set(o.keys())
    nk = set(n.keys())
    if ok != nk:
        print(f"\nLayer {idx} config key diff: old-only={ok-nk} new-only={nk-ok}")
    for k in sorted(ok & nk):
        if k in ("layer", "backward_layer"):
            continue
        if o[k] != n[k]:
            print(f"Layer {idx} config['{k}'] differs: old={str(o[k])[:120]} new={str(n[k])[:120]}")