import json

with open('C:/Users/BRENT/Downloads/handson_fixed/handson/normalize_test.json') as f:
    norm = json.load(f)
with open('C:/Users/BRENT/Downloads/handson_fixed/handson/training/tfjs_model_old_backup/model.json') as f:
    working = json.load(f)

n_layers = norm['modelTopology']['model_config']['config']['layers']
w_layers = working['modelTopology']['model_config']['config']['layers']

print('normalized layers:', len(n_layers))
print('working layers:', len(w_layers))
print()

for i, (n, w) in enumerate(zip(n_layers, w_layers)):
    if n != w:
        print(f'Layer {i} STILL DIFFERS: {n["class_name"]}')
        n_cfg = n.get('config', {})
        w_cfg = w.get('config', {})
        for k in set(n_cfg) | set(w_cfg):
            if n_cfg.get(k) != w_cfg.get(k):
                nv = str(n_cfg.get(k))[:200]
                wv = str(w_cfg.get(k))[:200]
                print(f'  config.{k}:')
                print(f'    NORM: {nv}')
                print(f'    WORK: {wv}')
        print()

# Check weight names
n_names = [w['name'] for w in norm['weightsManifest'][0]['weights']]
w_names = [w['name'] for w in working['weightsManifest'][0]['weights']]
print('weight name count norm:', len(n_names), 'work:', len(w_names))
print('weight names match:', n_names == w_names)
if n_names != w_names:
    for i, (a, b) in enumerate(zip(n_names, w_names)):
        if a != b:
            print(f'  weight {i}: NORM={a} WORK={b}')
            break