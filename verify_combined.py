import json
from collections import Counter

with open('training/combined_dataset.json') as f:
    data = json.load(f)

print(f'Samples: {len(data["samples"])}')
print(f'Labels: {len(data["labels"])}')
print(f'Feature dim: {data["featureDim"]}')
print(f'Window size: {data["windowSize"]}')
print(f'Stride: {data["stride"]}')

counts = Counter(s['label'] for s in data['samples'])
print(f'\nLabel distribution:')
for label, count in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'  {label}: {count}')