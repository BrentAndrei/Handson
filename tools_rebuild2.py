import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

OLD_JSON = os.path.join(base, "training", "tfjs_model_old_backup", "model.json")
NEW_BIN = os.path.join(base, "training", "model_out", "tfjs_model_399", "group1-shard1of1.bin")
OLD_BIN = os.path.join(base, "training", "tfjs_model_old_backup", "group1-shard1of1.bin")
OUT_JSON = os.path.join(base, "public", "models", "fsl_model", "model.json")
OUT_BIN = os.path.join(base, "public", "models", "fsl_model", "group1-shard1of1.bin")

with open(OLD_JSON) as f:
    m = json.load(f)

cfg = m["modelTopology"]["model_config"]["config"]
layers = cfg["layers"]

# 1. class_probs Dense: units 303 -> 399
for layer in layers:
    c = layer.get("config", {})
    if layer.get("name") == "class_probs":
        assert c["units"] == 303, c["units"]
        c["units"] = 399
        print("class_probs units -> 399")

# 2. weightsManifest: class_probs shapes 303 -> 399
wm = m["weightsManifest"]
for group in wm:
    for w in group.get("weights", []):
        if w["name"] == "class_probs/kernel":
            assert w["shape"] == [64, 303], w["shape"]
            w["shape"] = [64, 399]
            print("class_probs/kernel shape -> [64, 399]")
        elif w["name"] == "class_probs/bias":
            assert w["shape"] == [303], w["shape"]
            w["shape"] = [399]
            print("class_probs/bias shape -> [399]")

# 3. Verify bin layout matches manifest (float16 quantized, so 2 bytes per element)
with open(NEW_BIN, "rb") as f:
    new_bytes = f.read()

total = 0
for group in wm:
    for w in group.get("weights", []):
        size = 1
        for s in w["shape"]:
            size *= s
        total += size * 2  # float16 = 2 bytes
print(f"Manifest expects {total} bytes; bin has {len(new_bytes)} bytes")
assert total == len(new_bytes), f"BIN LAYOUT MISMATCH: {total} != {len(new_bytes)}"

# 4. Verify byte-prefix identity with old bin up to class_probs/kernel
with open(OLD_BIN, "rb") as f:
    old_bytes = f.read()

prefix_len = 0
for group in wm:
    for w in group.get("weights", []):
        if w["name"].startswith("class_probs"):
            break
        size = 1
        for s in w["shape"]:
            size *= s
        prefix_len += size * 2
    else:
        continue
    break

same = old_bytes[:prefix_len] == new_bytes[:prefix_len]
print(f"Prefix (non-class_probs, {prefix_len} bytes) identical to old bin: {same}")
if not same:
    n = 0
    for a, b in zip(old_bytes[:prefix_len], new_bytes[:prefix_len]):
        if a != b:
            break
        n += 1
    print(f"  First divergence at byte {n} of {prefix_len}")

with open(OUT_JSON, "w") as f:
    json.dump(m, f)
print(f"\nWrote {OUT_JSON}")

with open(OUT_BIN, "wb") as f:
    f.write(new_bytes)
print(f"Wrote {OUT_BIN} ({len(new_bytes)} bytes)")