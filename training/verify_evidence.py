import json, os, re
from pathlib import Path

BASE = Path(__file__).parent.parent

# Question 3: INFERENCE_LABELS count
print("=" * 60)
print("Q3: INFERENCE_LABELS count in App.tsx")
print("=" * 60)
content = (BASE / 'src' / 'App.tsx').read_text(encoding='utf-8')
lines = content.split('\n')
in_labels = False
inference_labels = []
for line in lines:
    if 'const INFERENCE_LABELS' in line:
        in_labels = True
        continue
    if in_labels:
        if '];' in line:
            break
        m = re.findall(r'"([^"]*)"', line)
        inference_labels.extend(m)
print(f"INFERENCE_LABELS count: {len(inference_labels)}")
print(f"First 3: {inference_labels[:3]}")
print(f"Last 3: {inference_labels[-3:]}")

# Check TFJS model output classes
print(f"\nTFJS model files:")
model_dir = BASE / 'public' / 'models' / 'fsl_model'
for f in sorted(model_dir.iterdir()):
    print(f"  {f.name}: {f.stat().st_size / 1024:.1f}KB")

# Check model output dimension from model.json
model_json = json.loads((model_dir / 'model.json').read_text())
for wg in model_json.get('weightsManifest', []):
    for wt in wg.get('weights', []):
        if 'class_probs' in wt.get('name', ''):
            print(f"\n  class_probs weight: {wt['name']}, shape: {wt['shape']}")
            print(f"  -> Output classes: {wt['shape'][-1]}")

# Question 1: Training output/logs
print("\n" + "=" * 60)
print("Q1: Training output and logs")
print("=" * 60)
training_dir = BASE / 'training'
log_files = list(training_dir.rglob("*.log"))
result_files = list(training_dir.rglob("*accuracy*"))
report_files = list(training_dir.rglob("*report*"))
print(f"Log files in training/: {len(log_files)}")
for lf in log_files[:5]:
    print(f"  {lf}")
print(f"Accuracy/report files: {len(result_files) + len(report_files)}")
for rf in result_files[:5]:
    print(f"  {rf}")
for rf in report_files[:5]:
    print(f"  {rf}")

# Check for any recent training artifacts
print("\nSearching for training artifacts...")
for p in training_dir.iterdir():
    print(f"  {p.name} ({p.stat().st_size / 1024:.0f}KB)" if p.is_file() else f"  {p.name}/")

# Question 2: Motion filter
print("\n" + "=" * 60)
print("Q2: Motion filtering pipeline")
print("=" * 60)
filter_script = training_dir / 'filter_low_motion_windows.py'
print(f"filter_low_motion_windows.py exists: {filter_script.exists()}")
if filter_script.exists():
    txt = filter_script.read_text()
    lines = txt.split('\n')
    for line in lines:
        if 'DATASET_PATH' in line:
            print(f"Input: {line.strip()[:100]}")
        if 'OUTPUT_PATH' in line:
            print(f"Output: {line.strip()[:100]}")
    filtered = training_dir / 'combined_dataset_filtered.json'
    print(f"filtered output exists: {filtered.exists()} (size: {filtered.stat().st_size / 1024 / 1024:.0f}MB)")

# Question 4: Avatar retargeting data
print("\n" + "=" * 60)
print("Q4: Avatar retargeting pipeline data coverage")
print("=" * 60)
retarget_file = BASE / 'src' / 'avatar' / 'retarget.ts'
print(f"retarget.ts exists: {retarget_file.exists()}")
if retarget_file.exists():
    print(f"retarget.ts lines: {len(retarget_file.read_text().splitlines())}")

avatar_dir = BASE / 'public' / 'avatar-data'
avatar_files = [f.stem for f in avatar_dir.glob('*.json') if f.stem != 'manifest']
print(f"Avatar data files: {len(avatar_files)}")
print(f"Manifest file exists: {(avatar_dir / 'manifest.json').exists()}")

manifest = json.loads((avatar_dir / 'manifest.json').read_text()) if (avatar_dir / 'manifest.json').exists() else {}
print(f"Manifest totalLabels: {manifest.get('totalLabels', 'N/A')}")
print(f"Manifest labels in list: {len(manifest.get('labels', []))}")

# How many inference labels have real avatar data (non-placeholder)?
real_data = 0
placeholder_data = 0
placeholder_labels = []
for label in inference_labels:
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', label)
    filepath = avatar_dir / f'{sanitized}.json'
    if filepath.exists():
        data = json.loads(filepath.read_text())
        if data.get('frameCount', 0) > 1 or len(data.get('frames', [])) > 345:
            real_data += 1
        else:
            placeholder_data += 1
            placeholder_labels.append(label)
    # else: no data at all

print(f"\nInference labels with REAL avatar data: {real_data}")
print(f"Inference labels with PLACEHOLDER avatar data: {placeholder_data}")
print(f"Inference labels with NO avatar data: {len(inference_labels) - real_data - placeholder_data}")
if placeholder_labels:
    print(f"Placeholder labels: {placeholder_labels[:10]}")

# Also check how many INFERENCE_LABELS via sanitized match
sanitized_match = sum(1 for l in inference_labels if re.sub(r'[^a-zA-Z0-9]', '_', l) in set(avatar_files))
print(f"\nTotal INFERENCE_LABELS with any avatar file (sanitized match): {sanitized_match}/{len(inference_labels)}")
