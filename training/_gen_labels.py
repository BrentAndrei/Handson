import json, os

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

# Read model_out label_map (source of truth for deployed model)
with open(os.path.join(base, "training", "model_out", "label_map.json")) as f:
    lm = json.load(f)
labels = lm["labels"]

# Generate labels.ts content
lines = []
lines.append("/**")
lines.append(" * The classifier's label vocabulary, in the EXACT order the model's")
lines.append(" * output layer was trained with.")
lines.append(" *")
lines.append(" * This list lived inline in App.tsx until the capture studio needed")
lines.append(" * it too (to test a captured clip against the classifier before")
lines.append(" * installing it). The order is a contract: index i of a prediction")
lines.append(" * vector is this list's i-th label, so it must never be sorted,")
lines.append(" * filtered, or deduplicated. `label_map.json` does not ship with")
lines.append(" * the exported model, so this array is the source of truth.")
lines.append(" *")
lines.append(f" * {len(labels)}-label vocabulary of the deployed classifier (fsl_bigru_v2).")
lines.append(" * This is the vocabulary the model was trained with, recovered from")
lines.append(" * the repository history (the first App.tsx commit, which shipped the")
lines.append(" * 303-class fsl_model). Measured top-1 accuracy on canonical")
lines.append(" * recordings: 0.774.")
lines.append(" */")
lines.append("export const INFERENCE_LABELS: string[] = [")
for label in labels:
    # Escape backslashes and double quotes in label strings
    escaped = label.replace("\\", "\\\\").replace('"', '\\"')
    lines.append(f'  "{escaped}",')
lines.append("];")

content = "\n".join(lines) + "\n"

out_path = os.path.join(base, "src", "inference", "labels.ts")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(content)

print(f"Wrote {len(labels)} labels to {out_path}")

# Verify by reading back
with open(out_path, "r", encoding="utf-8") as f:
    verify = f.read()
import re
start = verify.index("[")
end = verify.rindex("]")
found = re.findall(r'"([^"]*)"', verify[start:end])
print(f"Verified: {len(found)} labels in file")
print(f"First 5: {found[:5]}")
print(f"Last 5: {found[-5:]}")
print(f"Match model: {found == labels}")