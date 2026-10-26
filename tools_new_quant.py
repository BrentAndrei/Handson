import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

with open(os.path.join(base, "training", "model_out", "tfjs_model_399", "model.json")) as f:
    new = json.load(f)

print("NEW raw weight entry:")
print(json.dumps(new["weightsManifest"][0]["weights"][0], indent=2))
print("NEW quantization present:", "quantization" in new["weightsManifest"][0]["weights"][0])