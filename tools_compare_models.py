import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

# Compare working old model vs new model structure
with open(os.path.join(base, "training", "tfjs_model_old_backup", "model.json")) as f:
    old = json.load(f)
with open(os.path.join(base, "public", "models", "fsl_model", "model.json")) as f:
    new = json.load(f)

print("OLD model.json keys:", list(old.keys()))
print("NEW model.json keys:", list(new.keys()))

# Check modelTopology structure
old_topo = old.get("modelTopology", {})
new_topo = new.get("modelTopology", {})
print(f"\nOLD modelTopology keys: {list(old_topo.keys())}")
print(f"NEW modelTopology keys: {list(new_topo.keys())}")

# Check model_config structure
old_cfg = old_topo.get("model_config", {})
new_cfg = new_topo.get("model_config", {})
print(f"\nOLD model_config type: {type(old_cfg).__name__}")
print(f"NEW model_config type: {type(new_cfg).__name__}")

# Check InputLayer
old_layers = old_cfg.get("config", {}).get("layers", [])
new_layers = new_cfg.get("config", {}).get("layers", [])
print(f"\nOLD InputLayer config: {old_layers[0]['config'] if old_layers else 'N/A'}")
print(f"NEW InputLayer config: {new_layers[0]['config'] if new_layers else 'N/A'}")

# Check weightsManifest
print(f"\nOLD weightsManifest: {len(old.get('weightsManifest', []))} groups")
print(f"NEW weightsManifest: {len(new.get('weightsManifest', []))} groups")

# Check format
print(f"\nOLD format: {old.get('format')}")
print(f"NEW format: {new.get('format')}")