import json

with open('public/avatar-data/manifest.json') as f:
    m = json.load(f)

print(f'Total labels in manifest: {m["totalLabels"]}')
print(f'Labels: {m["labels"][:10]}...')