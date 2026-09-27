import json, os, re, sys
from pathlib import Path

BASE = Path(__file__).parent.parent

def get_inference_labels():
    app_file = BASE / 'src' / 'App.tsx'
    content = app_file.read_text(encoding='utf-8')
    start = content.find('INFERENCE_LABELS')
    if start == -1:
        print("ERROR: Could not find INFERENCE_LABELS in App.tsx")
        sys.exit(1)
    start = content.find('=', start) + 1
    start = content.find('[', start)
    end = content.find(']', start + 1)
    labels_str = content[start+1:end]
    labels = re.findall(r'["\']([^"\']+)["\']', labels_str)
    return labels

def get_avatar_labels():
    avatar_dir = BASE / 'public' / 'avatar-data'
    labels = set()
    for f in avatar_dir.glob('*.json'):
        labels.add(f.stem)
    return labels

def get_labels_from_json(filepath, max_bytes=300000):
    labels = set()
    all_labels_field = None
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read(max_bytes)
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return labels, False, None
        if isinstance(data, dict):
            if 'labels' in data and isinstance(data['labels'], list):
                for item in data['labels']:
                    if isinstance(item, str):
                        labels.add(item)
                all_labels_field = 'top-level labels'
            if 'samples' in data and isinstance(data['samples'], list):
                for item in data['samples'][:100]:
                    if isinstance(item, dict) and 'label' in item:
                        labels.add(item['label'])
                if all_labels_field is None:
                    all_labels_field = 'samples labels (first 100)'
        return labels, True, all_labels_field
    except Exception as e:
        print(f"  Error: {e}")
        return labels, False, None

def get_tfjs_model_labels():
    model_dir = BASE / 'public' / 'models' / 'fsl_model'
    model_json_path = model_dir / 'model.json'
    if not model_json_path.exists():
        return set(), 0
    try:
        with open(model_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        labels = set()
        if 'modelTopology' in data and isinstance(data['modelTopology'], dict):
            mt = data['modelTopology']
            if 'layers' in mt and isinstance(mt['layers'], list):
                for layer in mt['layers']:
                    if isinstance(layer, dict) and layer.get('class_name') == 'Dense':
                        config = layer.get('config', {})
                        if 'units' in config:
                            units = config['units']
                            labels.add(f"dense_{units}")
            if 'output_layers' in mt:
                ol = mt['output_layers']
                if isinstance(ol, list):
                    for o in ol:
                        if isinstance(o, dict):
                            if 'name' in o:
                                labels.add(o['name'])
                            if 'class_name' in o:
                                labels.add(o['class_name'])
        weights = data.get('weightsManifest', [])
        for w in weights:
            if isinstance(w, dict) and 'weights' in w:
                for wt in w['weights']:
                    if isinstance(wt, dict) and 'name' in wt:
                        shape = wt.get('shape', [])
                        if len(shape) >= 2:
                            labels.add(f"{wt['name']}_out{shape[-1]}")
        return labels, len(data.get('weightsManifest', []))
    except Exception as e:
        print(f"  Error reading model.json: {e}")
        return set(), 0

def sanitize(label):
    return ''.join(c if c.isalnum() else '_' for c in label)

def main():
    inference_labels = get_inference_labels()
    print(f"INFERENCE_LABELS (from App.tsx): {len(inference_labels)}")

    avatar_labels = get_avatar_labels()
    print(f"Avatar data files: {len(avatar_labels)}")

    inference_set = set(inference_labels)
    
    # Check via sanitized name matching (same as dataLoader.ts)
    avatar_sanitized = set()
    for f in avatar_labels:
        if f == 'manifest':
            continue
        avatar_sanitized.add(f)
    
    direct_match = inference_set & avatar_labels
    sanitized_match = sum(1 for l in inference_set if sanitize(l) in avatar_sanitized)
    truly_missing = [l for l in inference_set if sanitize(l) not in avatar_sanitized]
    
    print(f"\n--- Avatar Data Coverage ---")
    print(f"Direct name match: {len(direct_match)}/{len(inference_set)}")
    print(f"Sanitized name match: {sanitized_match}/{len(inference_set)}")
    print(f"Truly missing: {len(truly_missing)}")
    if truly_missing:
        print(f"  Missing labels: {truly_missing[:10]}")

    avatar_extra = avatar_labels - inference_set
    print(f"Extra avatar files (not in INFERENCE_LABELS): {len(avatar_extra)}")
    if avatar_extra:
        print(f"  Examples: {sorted(avatar_extra)[:10]}")

    print("\n--- TFJS Model ---")
    model_labels, num_weight_groups = get_tfjs_model_labels()
    print(f"TFJS model output classes: {len(model_labels)} (from {num_weight_groups} weight groups)")
    model_match = len(inference_set & model_labels)
    print(f"  INFERENCE_LABELS in model: {model_match}")
    print(f"  Model uses {len(model_labels)} named weight groups (output dim 328 from class_probs/kernel shape)")

    print("\n--- Large Datasets ---")
    training = BASE / 'training'
    large_files = [
        'combined_dataset_filtered.json',
        'fsl_dataset.json',
        'fsl_dataset_guiron_merged.json',
        'combined_dataset_filtered_v3.json',
        'combined_dataset_filtered_v4.json',
        'combined_dataset_compiled.json',
    ]
    all_dataset_labels = set()
    for fname in large_files:
        fp = training / fname
        if not fp.exists():
            print(f"  NOT FOUND: {fname}")
            continue
        size_mb = fp.stat().st_size / (1024*1024)
        labels, ok, label_src = get_labels_from_json(str(fp), max_bytes=300000)
        all_dataset_labels.update(labels)
        coverage = len(inference_set & labels)
        print(f"  {fname} ({size_mb:.0f}MB): {len(labels)} unique labels ({label_src}), {coverage} inference labels covered")

    print(f"\n--- Summary ---")
    print(f"Inference labels with avatar data (sanitized): {sanitized_match}/{len(inference_set)}")
    print(f"Avatar data files total: {len(avatar_labels)}")
    print(f"Still need real avatar data for: {len(truly_missing)} labels (may be placeholders)")

if __name__ == '__main__':
    main()
