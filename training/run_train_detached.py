"""Launcher: runs train.py with stdout/stderr redirected to files.
Use this instead of Start-Process so training output is captured.
Usage: python run_train_detached.py <dataset> <out_dir> <epochs>
"""
import subprocess
import sys
import os

dataset = sys.argv[1]
out_dir = sys.argv[2]
epochs = sys.argv[3] if len(sys.argv) > 3 else "300"

train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
log_out = os.path.join(out_dir, "training_merged.log")
log_err = os.path.join(out_dir, "training_merged.err")

os.makedirs(out_dir, exist_ok=True)

cmd = [
    sys.executable, train_py,
    "--dataset", dataset,
    "--out", out_dir,
    "--epochs", epochs,
]
print(f"Launching: {' '.join(cmd)}")
print(f"stdout -> {log_out}")
print(f"stderr -> {log_err}")

with open(log_out, "w", buffering=1) as out_f, open(log_err, "w", buffering=1) as err_f:
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f, text=True)
    print(f"Started PID {proc.pid}")
    proc.wait()
    print(f"Exit code: {proc.returncode}")