import json

with open('C:/Users/BRENT/Downloads/handson_fixed/handson/training/tfjs_model_old_backup/model.json') as f:
    working = json.load(f)
with open('C:/Users/BRENT/Downloads/handson_fixed/handson/public/models/fsl_model/model.json') as f:
    broken = json.load(f)

print('=== WORKING weightsManifest[0] ===')
print(json.dumps(working['weightsManifest'][0], indent=2)[:800])
print()
print('=== BROKEN weightsManifest[0] ===')
print(json.dumps(broken['weightsManifest'][0], indent=2)[:800])
print()
print('=== WORKING training_config ===')
print(json.dumps(working['modelTopology'].get('training_config'), indent=2)[:400])
print()
print('=== BROKEN training_config ===')
print(json.dumps(broken['modelTopology'].get('training_config'), indent=2)[:400])