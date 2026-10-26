import json

with open('C:/Users/BRENT/Downloads/handson_fixed/handson/training/tfjs_model_old_backup/model.json') as f:
    working = json.load(f)
with open('C:/Users/BRENT/Downloads/handson_fixed/handson/public/models/fsl_model/model.json') as f:
    broken = json.load(f)

w_layers = working['modelTopology']['model_config']['config']['layers']
b_layers = broken['modelTopology']['model_config']['config']['layers']

print('working layers:', len(w_layers))
print('broken layers:', len(b_layers))
print()
for i, (w, b) in enumerate(zip(w_layers, b_layers)):
    if w != b:
        print(f'Layer {i} DIFFERS: {w["class_name"]} / {b["class_name"]}')
        # Find first differing key
        w_keys = set(w.keys())
        b_keys = set(b.keys())
        if w_keys != b_keys:
            print('  key diff:', w_keys.symmetric_difference(b_keys))
        w_cfg = w.get('config', {})
        b_cfg = b.get('config', {})
        if w_cfg != b_cfg:
            for k in set(w_cfg) | set(b_cfg):
                if w_cfg.get(k) != b_cfg.get(k):
                    print(f'  config.{k}: WORKING={w_cfg.get(k)} BROKEN={b_cfg.get(k)}')
        print()