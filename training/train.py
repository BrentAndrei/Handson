"""
Trains a lightweight Bidirectional GRU sequence classifier on the
fsl_dataset.json exported by the "Collect data" mode of the React app
(src/features/datasetRecorder.ts).

Why Keras/TF and not PyTorch: the next pipeline step is converting to
TensorFlow.js. tensorflowjs_converter's --output_format=tfjs_layers_model
path re-implements Keras RNN layers (LSTM/GRU) directly in JS, which is
the reliable way to get recurrent layers running in-browser. Going via
PyTorch would mean PyTorch -> ONNX -> TensorFlow -> TFJS, and recurrent
ops are exactly the op family that tends to break across that many hops
(ONNX's LSTM/GRU op sets don't line up cleanly with TF's, and graph-level
conversion of RNN ops to TFJS's kernel set is unreliable). If you'd
rather stay in PyTorch: export to ONNX with dynamic axes on the time
dimension, convert ONNX -> TF SavedModel with onnx-tf, then follow the
exact same tensorflowjs_converter step in step_3_convert.sh — expect to
debug unsupported-op errors on the LSTM/GRU nodes specifically.

Usage:
    python train.py --dataset fsl_dataset.json --out ./model_out --epochs 60

Input dataset schema (see DatasetFile in datasetRecorder.ts):
{
  "featureDim": 345,
  "windowSize": 16,
  "labels": ["KUMUSTA", "SALAMAT", ...],
  "samples": [ { "label": "KUMUSTA", "frames": [<windowSize*featureDim floats>] }, ... ]
}
"""
import os
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import argparse
import json
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix


def load_dataset(path: str, consolidate: bool = False):
    with open(path, "r") as f:
        data = json.load(f)

    feature_dim = data["featureDim"]
    window_size = data["windowSize"]

    if consolidate:
        remap = {}
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            remap[ch] = "FINGERSPELL"
        remap["CONTOL"] = "CONTROL"
        remap["MAYBER"] = "MAYBE"
        for sample in data["samples"]:
            if sample["label"] in remap:
                sample["label"] = remap[sample["label"]]

    labels = sorted(set(s["label"] for s in data["samples"]))

    label_to_idx = {label: i for i, label in enumerate(labels)}

    X = np.zeros((len(data["samples"]), window_size, feature_dim), dtype=np.float32)
    y = np.zeros((len(data["samples"]),), dtype=np.int64)

    for i, sample in enumerate(data["samples"]):
        frames = np.asarray(sample["frames"], dtype=np.float32)
        expected = window_size * feature_dim
        if frames.size != expected:
            raise ValueError(
                f"Sample {i} has {frames.size} floats, expected {expected} "
                f"({window_size} frames x {feature_dim} features). "
                "If you're mixing in externally-recorded variable-length "
                "clips, pad/truncate them to window_size frames before "
                "adding them to the dataset — see pad_or_truncate() below "
                "for the same logic used at collection time in the app."
            )
        X[i] = frames.reshape(window_size, feature_dim)
        y[i] = label_to_idx[sample["label"]]

    # clip_id is written by extract_landmarks.py. Browser-recorded datasets
    # may not have it; give each of those independent fixed-size samples its
    # own group so this remains backwards compatible (and makes the fallback
    # explicit to the caller below).
    clip_ids = [sample.get("clip_id") for sample in data["samples"]]
    return X, y, labels, feature_dim, window_size, clip_ids


def split_by_clip(y: np.ndarray, clip_ids: list, val_split: float, seed: int):
    """Stratify by label while assigning every window from a clip together."""
    clips_by_class: dict[int, dict[str, list[int]]] = {}
    missing_clip_ids = 0
    for sample_idx, (label_idx, clip_id) in enumerate(zip(y, clip_ids)):
        if clip_id is None:
            # A legacy/browser sample has no known neighbouring windows.
            clip_id = f"legacy_window_{sample_idx}"
            missing_clip_ids += 1
        clips_by_class.setdefault(int(label_idx), {}).setdefault(str(clip_id), []).append(sample_idx)

    train_idx, val_idx = [], []
    rng = np.random.default_rng(seed)
    for clip_groups in clips_by_class.values():
        clip_keys = np.array(list(clip_groups))
        rng.shuffle(clip_keys)
        # Never leave a class with no training clips. A one-clip class cannot
        # be represented in both partitions and stays in training.
        n_val_clips = max(1, int(len(clip_keys) * val_split)) if len(clip_keys) > 1 else 0
        for clip_id in clip_keys[:n_val_clips]:
            val_idx.extend(clip_groups[clip_id])
        for clip_id in clip_keys[n_val_clips:]:
            train_idx.extend(clip_groups[clip_id])

    return np.array(train_idx), np.array(val_idx), missing_clip_ids


def pad_or_truncate(frames: np.ndarray, window_size: int) -> np.ndarray:
    """Pads (edge-repeat) or truncates a [T, feature_dim] clip to exactly
    window_size frames. Only needed if you're feeding in variable-length
    clips from a source other than the app's fixed-size window recorder —
    the app already emits fixed window_size windows, so this isn't called
    during normal training on fsl_dataset.json."""
    t = frames.shape[0]
    if t == window_size:
        return frames
    if t > window_size:
        start = (t - window_size) // 2
        return frames[start : start + window_size]
    pad_width = window_size - t
    pad_before = pad_width // 2
    pad_after = pad_width - pad_before
    return np.pad(frames, ((pad_before, pad_after), (0, 0)), mode="edge")


def build_model(window_size: int, feature_dim: int, num_classes: int, lr: float = 2e-4) -> tf.keras.Model:
    """Larger architecture for 100+ classes: wider GRUs, deeper classifier,
    and explicit L2 regularization on the Dense stack. The old 64->32 GRU
    with a 32-unit bottleneck was fine for ~100 classes but saturates fast
    above that — with 131+ classes you need roughly 3-4x the parameter budget
    in the classifier head or the model can't separate close classes."""
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
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def augment_window(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies cheap, landmark-preserving augmentations to a single
    [window_size, feature_dim] window. All ops are differentiable-through-
    identity (no interpolation), so the augmented sample is still a valid
    landmark vector the model can classify.

    1. Gaussian noise: tiny per-coordinate jitter so the model doesn't
       rely on exact landmark positions (which are themselves noisy).
    2. Time jitter: swap two adjacent frames with small probability. Breaks
       the model's dependence on exact frame-to-frame velocity, which is
       the main way it distinguishes signs in the first place.
    """
    out = window.copy()

    # 1. Gaussian noise
    out += rng.normal(0.0, 0.01, out.shape).astype(np.float32)

    # 2. Adjacent-frame swap
    if rng.random() < 0.1:
        i = rng.integers(0, window.shape[0] - 1)
        tmp = out[i].copy()
        out[i] = out[i + 1]
        out[i + 1] = tmp

    return out


class AugmentedSequence(tf.keras.utils.Sequence):
    """Fits the Keras `Sequence` interface so `model.fit` can use it as a
    custom data generator. On each epoch it re-shuffles and re-augments the
    training data; validation data is passed through untouched."""
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int = 16, seed: int = 42, augment: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, idx: int):
        batch = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        if self.augment:
            self.rng.shuffle(self.indices)
        x_batch = np.empty((len(batch), self.X.shape[1], self.X.shape[2]), dtype=np.float32)
        for i, b in enumerate(batch):
            x_batch[i] = self.X[b] if not self.augment else augment_window(self.X[b], self.rng)
        return x_batch, self.y[batch]

    def on_epoch_end(self):
        if self.augment:
            self.rng.shuffle(self.indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to fsl_dataset.json")
    parser.add_argument("--out", default="./model_out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-split", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate for Adam optimizer")
    parser.add_argument("--consolidate", action="store_true", help="Consolidate labels: A-Z→FINGERSPELL, fix typos (CONTOL→CONTROL, MAYBER→MAYBE)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--random-split", action="store_true", help="Use random train/val split instead of clip-level split. Use this when you have few recording sessions — the clip-level split puts entire unseen sessions in validation, which with only ~11 sessions makes the val set tiny and unrepresentative. Random split tests recognition of held-out windows from sessions the model DID see, which is the honest metric for 'can this model recognize this signer's signs'.")
    args = parser.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    X, y, labels, feature_dim, window_size, clip_ids = load_dataset(args.dataset, consolidate=args.consolidate)
    counts = Counter(y.tolist())
    print(f"Loaded {len(X)} windows across {len(labels)} labels: {labels}")
    print(f"Per-class window counts: { {labels[k]: v for k, v in counts.items()} }")
    if min(counts.values()) < 10:
        print(
            "WARNING: at least one class has fewer than 10 windows. Sequence "
            "models overfit fast on small class-imbalanced sets like this — "
            "record more takes per sign (each take yields several windows "
            "via the stride) before trusting the val accuracy below."
        )

    if args.random_split:
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(X))
        rng.shuffle(indices)
        n_val = int(len(X) * args.val_split)
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]
        missing_clip_ids = 0
        clip_count = 0
        print(f"Random split: {len(train_idx)} train, {len(val_idx)} val windows")
    else:
        train_idx, val_idx, missing_clip_ids = split_by_clip(y, clip_ids, args.val_split, args.seed)
        clip_count = len({clip_id for clip_id in clip_ids if clip_id is not None})
        print(f"Clip-level split: {clip_count} identified clips; {len(train_idx)} train windows, {len(val_idx)} val windows")
        if missing_clip_ids:
            print(f"WARNING: {missing_clip_ids} legacy samples lack clip_id and were treated as independent clips. Re-extract video datasets for a strict clip-level split.")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Class weights: upsample rare classes so the model doesn't just learn
    # to predict the frequent classes. Inverse-frequency weighting.
    class_counts = np.bincount(y_train, minlength=len(labels))
    class_weights = np.where(class_counts > 0,
                             len(y_train) / (len(labels) * class_counts),
                             0.0).astype(np.float32)
    class_weights_dict = {i: float(w) for i, w in enumerate(class_weights)}

    model = build_model(window_size, feature_dim, len(labels), lr=args.lr)
    model.summary()

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "best.weights.h5")

    train_seq = AugmentedSequence(X_train, y_train, batch_size=args.batch_size, seed=args.seed, augment=not args.no_augment)
    val_seq = AugmentedSequence(X_val, y_val, batch_size=args.batch_size, seed=args.seed, augment=False)

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        class_weight=class_weights_dict,
    callbacks=[
        callbacks.ModelCheckpoint(
            ckpt_path, save_best_only=True, save_weights_only=True,
            monitor="val_accuracy" if len(X_val) > 0 else "accuracy",
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy" if len(X_val) > 0 else "accuracy",
            mode="max",
            patience=25,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss" if len(X_val) > 0 else "loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ],
    )

    keras_path = os.path.join(args.out, "fsl_model.h5")
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")

    label_map_path = os.path.join(args.out, "label_map.json")
    with open(label_map_path, "w") as f:
        json.dump({"labels": labels, "featureDim": feature_dim, "windowSize": window_size}, f, indent=2)
    print(f"Saved label map to {label_map_path}")
    print(
        "Copy `labels` from label_map.json into INFERENCE_LABELS in App.tsx, "
        "in this exact order — it must match the model's output index order."
    )

    if len(X_val) > 0:
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"Final val accuracy: {val_acc:.3f} (loss {val_loss:.3f})")

        y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
        print("\n=== Classification report ===")
        print(classification_report(y_val, y_pred, target_names=labels))

        cm = confusion_matrix(y_val, y_pred, labels=list(range(len(labels))))
        print("\n=== Per-class top-2 confusion ===")
        for i, lbl in enumerate(labels):
            row = cm[i]
            row[i] = 0
            top = np.argsort(row)[-2:][::-1]
            parts = []
            for j in top:
                if row[j] > 0:
                    parts.append(f"{labels[j]} ({row[j]} times)")
            if parts:
                print(f"{lbl} confused with: {', '.join(parts)}")
            else:
                print(f"{lbl} confused with: none")


if __name__ == "__main__":
    main()
