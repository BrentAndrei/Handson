import json, os

model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "public", "models", "fsl_model", "model.json")

with open(model_path) as f:
    content = f.read()

# Check if it's valid JSON
try:
    m = json.loads(content)
    print("Valid JSON")
except Exception as e:
    print(f"Invalid JSON: {e}")

# Check signature section
sig = m.get("signature", {})
print(f"Signature keys: {list(sig.keys()) if sig else 'none'}")
if sig:
    for k, v in sig.items():
        print(f"  {k}: type={type(v).__name__}")

# Check if there are any functions/objects that shouldn't be there
modelTopo = m.get("modelTopology", {})
print(f"\nmodelTopology type: {type(modelTopo).__name__}")

# Check if serialized is string (should be for older TFJS versions)
serialized = modelTopo.get("serialized")
print(f"serialized type: {type(serialized).__name__}")

# Print first 500 chars of the JSON
print(f"\nFirst 500 chars:\n{content[:500]}")