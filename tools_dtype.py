import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

paths = {
    "old_backup": os.path.join(base, "training", "tfjs_model_old_backup", "model.json"),
    "new_399": os.path.join(base, "training", "model_out", "tfjs_model_399", "model.json"),
}
bins = {
    "old_backup": os.path.join(base, "training", "tfjs_model_old_backup", "group1-shard1of1.bin"),
    "new_399": os.path.join(base, "training", "model_out", "tfjs_model_399", "group1-shard1of1.bin"),
}

for name, p in paths.items():
    with open(p) as f:
        m = json.load(f)
    print(f"=== {name} ===")
    total = 0
    for g in m.get("weightsManifest", []):
        for w in g.get("weights", []):
            shape = w.get("shape", [])
            size = 1
            for s in shape:
                size *= s
            dtype = w.get("dtype", "float32")
            bytes_per = 2 if dtype == "float16" else 4
            total += size * bytes_per
            print(f"  {w['name']}: {shape} dtype={dtype}")
    print(f"  manifest total bytes: {total}, bin size: {os.path.getsize(bins[name])}")
    print(f"  quantization: {m.get('weightsManifest', [{}])[0].get('weights', [{}])[0].get('dtype')}")
    print()