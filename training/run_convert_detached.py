"""Detached launcher for tensorflowjs_converter.
Runs the converter with stdout/stderr redirected to files so the
shell tool's timeout can't kill it mid-conversion.
"""
import subprocess
import sys
import os

model_in = sys.argv[1]
model_out = sys.argv[2]
log_out = model_out + ".convert.log"
log_err = model_out + ".convert.err"

# Use the installed console-script entry point directly
converter = os.path.join(os.path.dirname(sys.executable), "tensorflowjs_converter.exe")
cmd = [
    converter,
    "--input_format=keras",
    "--output_format=tfjs_layers_model",
    "--quantize_float16=*",
    model_in,
    model_out,
]
print(f"Launching: {' '.join(cmd)}")
print(f"stdout -> {log_out}")
print(f"stderr -> {log_err}")

os.makedirs(model_out, exist_ok=True)
with open(log_out, "w") as out_f, open(log_err, "w") as err_f:
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f)
    print(f"Started PID {proc.pid}")
    proc.wait()
    print(f"Exit code: {proc.returncode}")