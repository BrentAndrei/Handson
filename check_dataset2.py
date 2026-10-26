import json
with open('training/video_dataset.json') as f:
    data = json.load(f)
print(f'Samples: {len(data.get("samples", []))}')
print(f'Labels: {data.get("labels", [])}')
print(f'Feature dim: {data.get("featureDim")}')
print(f'Window size: {data.get("windowSize")}')
print(f'Stride: {data.get("stride")}')

from collections import Counter
label_counts = Counter(s['label'] for s in data.get('samples', []))
print(f'Samples per label: {dict(label_counts)}')

if 'samples' in data and data['samples']:
    sample = data['samples'][0]
    print(f'First sample keys: {list(sample.keys())}')
    print(f'Label: {sample.get("label")}')
    print(f'Frames (window) length: {len(sample.get("frames", []))} floats')
    expected = data.get('windowSize', 16) * data.get('featureDim', 345)
    print(f'Expected: {expected} floats (16 frames * 345 dims)')
    
    # Check a few more
    for i, s in enumerate(data['samples'][:5]):
        print(f'Sample {i}: label={s.get("label")}, clip_id={s.get("clip_id")}, frames={len(s.get("frames", []))}')