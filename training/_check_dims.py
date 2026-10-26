import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

candidates = {
    "deployed public/models/fsl_model": os.path.join(base, "public", "models", "fsl_model", "model.json"),
    "training/tfjs_model_old_backup": os.path.join(base, "training", "tfjs_model_old_backup", "model.json"),
    "training/model_out/tfjs_model_399": os.path.join(base, "training", "model_out", "tfjs_model_399", "model.json"),
    "training/train_out_merged/tfjs_model_merged": os.path.join(base, "training", "train_out_merged", "tfjs_model_merged", "model.json"),
}

for name, path in candidates.items():
    if not os.path.exists(path):
        print(f"{name}: NOT FOUND")
        continue
    with open(path) as f:
        m = json.load(f)
    cfg = m.get("modelTopology", {}).get("model_config", {}).get("config", {})
    layers = cfg.get("layers", [])
    out_units = None
    in_shape = None
    for L in layers:
        c = L.get("config", {})
        if L.get("class_name") == "InputLayer":
            in_shape = c.get("batch_input_shape")
        # final Dense with softmax
        if c.get("activation") == "softmax" and c.get("units"):
            out_units = c["units"]
    # weights manifest totals
    wm = m.get("weightsManifest", [])
    total_params = 0
    for group in wm:
        for w in group.get("weights", []):
            total_params += 1
    print(f"{name}:")
    print(f"    input shape : {in_shape}")
    print(f"    output units: {out_units}")
    print(f"    weight arrays: {total_params}")
    print(f"    model name  : {cfg.get('name')}")
