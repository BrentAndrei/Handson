import json, os, re
from pathlib import Path

BASE = Path(__file__).parent.parent

print("=" * 60)
print("ANSWERING USER'S FOUR QUESTIONS")
print("=" * 60)

# Q1: Raw training output for 303-label model
print("\n--- Q1: Training Output ---")
# Check training logs
training_dir = BASE / 'training'
logs = ['training_stdout.log', 'training_log.txt', 'reset_after_train.stdout.log']
for log in logs:
    path = training_dir / log
    if path.exists():
        content = path.read_text(encoding='utf-8', errors='ignore')
        # Find label counts mentioned
        label_matches = re.findall(r'Loaded\s+(\d+)\s+windows\s+across\s+(\d+)\s+labels', content)
        for count, label_count in label_matches:
            print(f"  {log}: {label_count} labels, {count} windows")
        
        # Find val accuracy
        val_acc = re.findall(r'val_accuracy:\s*([\d.]+)', content)
        if val_acc:
            print(f"  {log}: val_accuracy values found: {val_acc[-5:]}")
        
        final_val = re.findall(r'Final val accuracy:\s*([\d.]+)', content)
        if final_val:
            print(f"  {log}: Final val accuracy: {final_val}")
        
        # Find classification reports
        if 'Classification report' in content or 'precision' in content:
            class_count = len(re.findall(r'\s+[A-Z][A-Z]', content))
            print(f"  {log}: Contains classification report")

# Check model_compiled_consolidated label_map
lm = json.load(open(training_dir / 'model_compiled_consolidated' / 'label_map.json'))
labels = lm['labels']
print(f"\n  model_compiled_consolidated label_map: {len(labels)} labels")
print(f"  First 5: {labels[:5]}")
print(f"  Last 5: {labels[-5:]}")

# Check if there's a classification report for this model
consolidated_model_dir = training_dir / 'model_compiled_consolidated'
for f in consolidated_model_dir.iterdir():
    print(f"  File in model_compiled_consolidated: {f.name}")

# Q2: Motion filtering
print("\n--- Q2: Motion Filtering ---")
filter_script = training_dir / 'filter_low_motion_windows.py'
if filter_script.exists():
    txt = filter_script.read_text()
    print(f"  Script exists: Yes")
    print(f"  Input: combined_dataset.json")
    print(f"  Output: combined_dataset_filtered.json")
    filtered = training_dir / 'combined_dataset_filtered.json'
    print(f"  Output exists: {filtered.exists()} (size: {filtered.stat().st_size / 1024 / 1024:.0f}MB)")
    
    # Check labels in filtered dataset
    with open(filtered, 'r', encoding='utf-8') as f:
        header = f.read(200000)
    try:
        data = json.loads(header[:50000])
        if 'labels' in data:
            print(f"  Filtered dataset labels: {len(data['labels'])}")
            print(f"  Examples: {data['labels'][:5]}")
    except:
        print(f"  Could not parse labels from header")
else:
    print(f"  Script does NOT exist")

# Q3: INFERENCE_LABELS and TFJS model
print("\n--- Q3: INFERENCE_LABELS and TFJS Model ---")
content = (BASE / 'src' / 'App.tsx').read_text()
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
print(f"  INFERENCE_LABELS count in App.tsx: {len(inference_labels)}")
print(f"  First 3: {inference_labels[:3]}")
print(f"  Last 3: {inference_labels[-3:]}")

# Check TFJS model
model_dir = BASE / 'public' / 'models' / 'fsl_model'
model_json = json.loads((model_dir / 'model.json').read_text())
for wg in model_json.get('weightsManifest', []):
    for wt in wg.get('weights', []):
        if 'class_probs' in wt.get('name', ''):
            print(f"  TFJS model class_probs output: {wt['shape'][-1]} classes")
            break

# Check if INFERENCE_LABELS matches model_compiled_consolidated
lm_consolidated = json.load(open(training_dir / 'model_compiled_consolidated' / 'label_map.json'))
consolidated_labels = lm_consolidated['labels']
print(f"  model_compiled_consolidated labels: {len(consolidated_labels)}")
match = set(inference_labels) == set(consolidated_labels)
print(f"  INFERENCE_LABELS matches consolidated labels: {match}")
if not match:
    in_not_in = set(inference_labels) - set(consolidated_labels)
    con_not_in = set(consolidated_labels) - set(inference_labels)
    print(f"  In INFERENCE but not consolidated: {in_not_in}")
    print(f"  In consolidated but not INFERENCE: {con_not_in}")

# Q4: Avatar retargeting pipeline data coverage
print("\n--- Q4: Avatar Retargeting Data ---")
avatar_dir = BASE / 'public' / 'avatar-data'
avatar_files = set(f.stem for f in avatar_dir.glob('*.json') if f.stem != 'manifest')

real_data = 0
placeholder = 0
missing = []
for label in inference_labels:
    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', label)
    filepath = avatar_dir / f'{sanitized}.json'
    if filepath.exists():
        data = json.loads(filepath.read_text())
        if data.get('frameCount', 0) > 1 or len(data.get('frames', [])) > 345:
            real_data += 1
        else:
            placeholder += 1
    else:
        missing.append(label)

print(f"  Real avatar data: {real_data}/{len(inference_labels)}")
print(f"  Placeholder avatar data: {placeholder}/{len(inference_labels)}")
print(f"  Missing avatar data: {len(missing)}/{len(inference_labels)}")
if placeholder:
    p_labels = []
    for l in inference_labels:
        s = re.sub(r'[^a-zA-Z0-9]', '_', l)
        if s in avatar_files:
            try:
                d = json.loads((avatar_dir / f'{s}.json').read_text())
                if d.get('frameCount', 0) <= 1:
                    p_labels.append(l)
            except:
                pass
    print(f"  Placeholder labels: {p_labels[:10]}")
if missing:
    print(f"  Missing labels: {missing[:10]}")

# Check retarget.ts references
retarget = BASE / 'src' / 'avatar' / 'retarget.ts'
print(f"  retarget.ts exists: {retarget.exists()}, lines: {len(retarget.read_text().splitlines()) if retarget.exists() else 0}")
