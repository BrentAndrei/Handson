import json, os

for root, dirs, files in os.walk('C:/Users/BRENT/Downloads/handson_fixed/handson'):
    if 'node_modules' in root:
        continue
    for f in files:
        if f == 'model.json':
            fp = os.path.join(root, f)
            try:
                with open(fp) as fh:
                    d = json.load(fh)
                layers = d['modelTopology']['model_config']['config']['layers']
                last = [L for L in layers if L['class_name'] == 'Dense'][-1]
                bis = layers[0]['config'].get('batch_input_shape') or layers[0]['config'].get('batch_shape')
                print('converter=' + str(d.get('convertedBy')) + ' units=' + str(last['config'].get('units')) + ' shape=' + str(bis) + ' size=' + str(os.path.getsize(fp)) + ' ' + os.path.relpath(fp, 'C:/Users/BRENT/Downloads/handson_fixed/handson'))
            except Exception as e:
                pass