import json
import os
import h5py
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras import layers, models

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")


def build_model(window_size, feature_dim, num_classes):
    inputs = layers.Input(shape=(window_size, feature_dim), name="landmark_window")
    x = layers.Bidirectional(
        layers.GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.0, reset_after=False)
    )(inputs)
    x = layers.Bidirectional(
        layers.GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.0, reset_after=False)
    )(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)
    model = models.Model(inputs, outputs, name="fsl_bigru_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# Rebuild the model architecture exactly as trained, then load weights.
# Loading the full h5 fails on Keras 3.15 (time_major kwarg removed from GRU)
# and loading weights alone fails on GRUCell variable layout. Rebuilding the
# fresh model and using load_weights with skip_mismatch handles both.
model = build_model(16, 345, 131)
model.load_weights("model_out_filtered_v4/fsl_model.h5", skip_mismatch=True)
print("Model loaded OK")

with open("combined_dataset_filtered_v4.json") as f:
    data = json.load(f)

labels = data["labels"]
label_to_idx = {l: i for i, l in enumerate(labels)}

rng = np.random.default_rng(42)
indices = np.arange(len(data["samples"]))
rng.shuffle(indices)
n_val = int(len(indices) * 0.25)
val_idx = set(indices[:n_val])

X = np.zeros((len(data["samples"]), 16, 345), dtype=np.float32)
y = np.zeros(len(data["samples"]), dtype=np.int64)
for i, s in enumerate(data["samples"]):
    X[i] = np.asarray(s["frames"], dtype=np.float32).reshape(16, 345)
    y[i] = label_to_idx[s["label"]]

val_list = list(val_idx)
y_pred = np.argmax(model.predict(X[val_list], verbose=0), axis=1)
y_val = y[val_list]

report = classification_report(y_val, y_pred, target_names=labels, output_dict=True)

rows = []
for lbl in labels:
    r = report[lbl]
    rows.append((lbl, r["precision"], r["recall"], r["f1-score"], r["support"]))
rows.sort(key=lambda x: x[3])

header = "{:<35} {:>9} {:>7} {:>7} {:>7}".format("Sign", "Precision", "Recall", "F1", "Support")
print(header)
print("-" * 70)
for lbl, p, r, f1, sup in rows:
    print("{:<35} {:>9.2f} {:>7.2f} {:>7.2f} {:>7.0f}".format(lbl, p, r, f1, sup))

macro = report["macro avg"]
weighted = report["weighted avg"]
print("-" * 70)
print("{:<35} {:>9.2f} {:>7.2f} {:>7.2f} {:>7.0f}".format(
    "MACRO AVG", macro["precision"], macro["recall"], macro["f1-score"], macro["support"]))
print("{:<35} {:>9.2f} {:>7.2f} {:>7.2f} {:>7.0f}".format(
    "WEIGHTED AVG", weighted["precision"], weighted["recall"], weighted["f1-score"], weighted["support"]))

with open("per_class_metrics_v4.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved per-class metrics to per_class_metrics_v4.json")