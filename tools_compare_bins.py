import json, os, struct

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

# Old working model (303-class)
with open(os.path.join(base, "training", "tfjs_model_old_backup", "model.json")) as f:
    old = json.load(f)

# New model weights info (399-class) - from tfjs_model_399 model.json
with open(os.path.join(base, "training", "model_out", "tfjs_model_399", "model.json")) as f:
    new = json.load(f)

print("=== OLD weightsManifest ===")
for g in old.get("weightsManifest", []):
    for w in g.get("weights", []):
        shape = w.get("shape", [])
        size = 1
        for s in shape: size *= s
        print(f"  {w['name']}: shape={shape} size={size} bytes={size*4}")

print("\n=== NEW weightsManifest ===")
for g in new.get("weightsManifest", []):
    for w in g.get("weights", []):
        shape = w.get("shape", [])
        size = 1
        for s in shape: size *= s
        print(f"  {w['name']}: shape={shape} size={size} bytes={size*4}")

# Check bin file sizes
old_bin = os.path.join(base, "training", "tfjs_model_old_backup", "group1-shard1of1.bin")
new_bin = os.path.join(base, "training", "model_out", "tfjs_model_399", "group1-shard1of1.bin")
print(f"\nOld bin size: {os.path.getsize(old_bin)} bytes")
print(f"New bin size: {os.path.getsize(new_bin)} bytes")
print(f"Difference: {os.path.getsize(new_bin) - os.path.getsize(old_bin)} bytes")