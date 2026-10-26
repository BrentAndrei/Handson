import subprocess, os, sys

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"
h5_path = os.path.join(base, "training", "model_out", "fsl_model.h5")
output_dir = os.path.join(base, "public", "models", "fsl_model_new")

# Remove existing output dir
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)

# Run converter v3.18.0 from pip (no-deps install)
result = subprocess.run([
    sys.executable, "-m", "tensorflowjs_converter",
    "--output_format", "tfjs_layers_model",
    h5_path,
    output_dir
], capture_output=True, text=True, timeout=120)

print("STDOUT:", result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
print("STDERR:", result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
print("Return code:", result.returncode)