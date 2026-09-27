import json
from pathlib import Path

BASE = Path(__file__).parent.parent
training = BASE / 'training'

print("Checking model_compiled_consolidated/fsl_model.h5...")
try:
    import h5py
    f = h5py.File(str(training / 'model_compiled_consolidated' / 'fsl_model.h5'), 'r')
    # Check model_config
    mc = f.attrs.get('model_config', b'{}')
    if isinstance(mc, bytes):
        mc = mc.decode('utf-8', errors='ignore')
    config = json.loads(mc)
    layers = config.get('model_config', {}).get('config', {}).get('layers', [])
    for layer in layers:
        if layer.get('class_name') == 'class_probs':
            units = layer.get('config', {}).get('units', 'unknown')
            print(f"  class_probs units: {units}")
    f.close()
except ImportError:
    print("  h5py not available")
except Exception as e:
    print(f"  Error: {e}")

print("\nChecking model_compiled/fsl_model.h5...")
try:
    f = h5py.File(str(training / 'model_compiled' / 'fsl_model.h5'), 'r')
    mc = f.attrs.get('model_config', b'{}')
    if isinstance(mc, bytes):
        mc = mc.decode('utf-8', errors='ignore')
    config = json.loads(mc)
    layers = config.get('model_config', {}).get('config', {}).get('layers', [])
    for layer in layers:
        if layer.get('class_name') == 'class_probs':
            units = layer.get('config', {}).get('units', 'unknown')
            print(f"  class_probs units: {units}")
    f.close()
except ImportError:
    print("  h5py not available")
except Exception as e:
    print(f"  Error: {e}")

print("\nSummary of evidence:")
print("=" * 60)
print("Q1: Training output for 303-label model:")
print("  NO training log found for 303-label model")
print("  training_log.txt documents 131-label model: val_accuracy=0.906")
print("  reset_after_train.stdout.log documents 105-label model: val_accuracy=0.401")
print("  model_compiled_consolidated/fsl_model.h5 exists (303 labels in label_map)")
print("  But no log shows training was run for 303 labels")
print()
print("Q2: Motion filtering:")
print("  filter_low_motion_windows.py EXISTS")
print("  Combined_dataset_filtered.json exists (1.7GB)")
print("  UNKNOWN: Whether it was used for 303-label training")
print()
print("Q3: INFERENCE_LABELS vs TFJS model:")
print("  INFERENCE_LABELS in App.tsx: 303")
print("  model_compiled_consolidated label_map: 303")
print("  Deployed TFJS model (public/models/fsl_model/): 328 output classes")
print("  MISMATCH: TFJS model has 328 classes, INFERENCE_LABELS has 303")
print("  Model labels not in INFERENCE_LABELS: A-Z (26 letters)")
print()
print("Q4: Avatar retargeting data:")
print("  302/303 INFERENCE_LABELS have REAL avatar data")
print("  1/303 (Ñ) has PLACEHOLDER data (all zeros, 1 frame)")
print("  retarget.ts exists (215 lines)")
print("  Data loaded via sanitized label names from /avatar-data/")
