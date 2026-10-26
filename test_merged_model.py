"""Test train_out_merged/fsl_model.h5 (399 classes, the 'merged' model)
on the same canonical windows. Also compares its weights against the
deployed model_out/fsl_model.h5 to see which one is actually live.
"""
import json
import os

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

ROOT = "C:/Users/BRENT/Downloads/handson_fixed/handson"
DATA = f"{ROOT}/dist/avatar-data/HELLO.json"
WINDOW = 16
STRIDE = 3


def load_windows():
    with open(DATA) as f:
        data = json.load(f)
    frames = np.asarray(data["frames"], dtype=np.float32)
    fc = frames.shape[0] // 345
    frames = frames[: fc * 345].reshape(fc, 345)
    out = []
    for s in range(0, fc - WINDOW + 1, STRIDE):
        out.append(frames[s : s + WINDOW])
    return np.stack(out)


def report(name, model, windows, labels, n_classes):
    probs = model.predict(windows, verbose=0)
    tops = probs.max(axis=1)
    argmax = probs.argmax(axis=1)
    idx = labels.index("HELLO")
    correct = int((argmax == idx).sum())
    in_top3 = int((np.argsort(probs, axis=1)[:, -3:] == idx).any(axis=1).sum())
    print(f"\n{name}: classes={n_classes}")
    print(f"  HELLO idx={idx}  top1={correct}/{len(windows)} ({correct/len(windows):.0%})  "
          f"top3={in_top3/len(windows):.0%}  mean={tops.mean():.4f}  max={tops.max():.4f}")
    counts = {}
    for a in argmax.tolist():
        counts[a] = counts.get(a, 0) + 1
    sc = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print(f"  argmax top3: " + ", ".join(f"class_{k}:{v}" for k, v in sc))


def main():
    windows = load_windows()
    print(f"HELLO: {windows.shape[0]} windows")

    merged = tf.keras.models.load_model(f"{ROOT}/training/train_out_merged/fsl_model.h5")
    with open(f"{ROOT}/training/train_out_merged/label_map.json") as f:
        mlm = json.load(f)
    report("train_out_merged/fsl_model.h5", merged, windows, mlm["labels"], len(mlm["labels"]))

    base = tf.keras.models.load_model(f"{ROOT}/training/model_out/fsl_model.h5")
    with open(f"{ROOT}/training/model_out/label_map.json") as f:
        blm = json.load(f)
    report("model_out/fsl_model.h5 (deployed source)", base, windows, blm["labels"], len(blm["labels"]))

    # Are the two 399-class weight sets the same?
    a = base.get_weights()
    b = merged.get_weights()
    same = all(np.array_equal(x, y) for x, y in zip(a, b))
    print(f"\nmodel_out == train_out_merged weights? {same}")
    if not same:
        print("  max abs diff per layer:",
              [round(float(np.abs(x - y).max()), 5) for x, y in zip(a, b)])


if __name__ == "__main__":
    main()