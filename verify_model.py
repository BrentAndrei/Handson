import json, re, os

with open('C:/Users/BRENT/Downloads/handson_fixed/handson/public/models/fsl_model/model.json') as f:
    d = json.load(f)
layers = d['modelTopology']['model_config']['config']['layers']
last = [L for L in layers if L['class_name'] == 'Dense'][-1]
print('DEPLOYED output units:', last['config'].get('units'), 'name:', last['config'].get('name'))
bis = layers[0]['config'].get('batch_input_shape') or layers[0]['config'].get('batch_shape')
print('DEPLOYED input shape:', bis)

with open('C:/Users/BRENT/Downloads/handson_fixed/handson/src/inference/labels.ts') as f:
    content = f.read()
labels = re.findall(r'"([^"]+)"', content)
print('LABELS count:', len(labels))

for p in [
    'C:/Users/BRENT/Downloads/handson_fixed/handson/training/label_map.json',
    'C:/Users/BRENT/Downloads/handson_fixed/handson/training/model_out/label_map.json',
    'C:/Users/BRENT/Downloads/handson_fixed/handson/model_out/label_map.json',
]:
    if os.path.exists(p):
        with open(p) as f:
            lm = json.load(f)
        print('label_map', p, 'count:', len(lm))
        print('MATCH:', len(labels) == last['config'].get('units') == len(lm))
        break