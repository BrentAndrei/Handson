import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

with open(os.path.join(base, "training", "tfjs_model_old_backup", "model.json")) as f:
    old = json.load(f)

print("OLD raw weight entry:")
print(json.dumps(old["weightsManifest"][0]["weights"][0], indent=2))
print()
print("OLD weightsManifest group keys:", list(old["weightsManifest"][0].keys()))
print("OLD manifest top-level extra keys:", [k for k in old if k not in ("format","generatedBy","convertedBy","modelTopology","weightsManifest")])

# Check the actual training_config and modelTopology for "build_config"/"keras_version" hints
print("\nOLD training_config keys:", list(old["modelTopology"].get("training_config", {}).keys()))
tc = old["modelTopology"].get("training_config", {})
print("OLD training_config:", json.dumps(tc)[:1500])