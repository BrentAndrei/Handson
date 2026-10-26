import json, os, numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

base = os.getcwd()
model = tf.keras.models.load_model(os.path.join(base, 'training', 'model_out', 'fsl_model.h5'), compile=False)
with open(os.path.join(base, 'training', 'model_out', 'label_map.json')) as f:
    lm = json.load(f)
labels = lm['labels']
print('Num classes:', len(labels))

dummy = np.zeros((1, 16, 345), dtype=np.float32)
probs = model.predict(dummy, verbose=0)[0]
top5 = sorted(range(len(probs)), key=lambda i: probs[i])[-5:][::-1]
print('Top 5 on zero input:')
for i in top5:
    print(f'  {probs[i]:.4f} class_{i}')
print('Sum of all 399 probs:', sum(probs))
print('Max prob:', max(probs))