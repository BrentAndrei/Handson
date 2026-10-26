import subprocess, os, sys

base = r"C:\Users\BRENT\Downloads\handson_fixed\handson"

# Try to find tensorflowjs_converter
result = subprocess.run([sys.executable, "-m", "pip", "show", "-f", "tensorflowjs"],
                        capture_output=True, text=True, timeout=30)
lines = result.stdout.split("\n")
for line in lines:
    if "converter" in line.lower() or "bin" in line.lower() or "Location" in line:
        print(line)

# Check if tensorflowjs_converter.exe exists
site_packages = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "Lib", "site-packages")
converter = os.path.join(site_packages, "tensorflowjs_converter.exe")
print(f"\nConverter at {converter}: exists={os.path.exists(converter)}")

# List scripts
scripts_dir = os.path.join(site_packages, "Scripts")
if os.path.exists(scripts_dir):
    for f in os.listdir(scripts_dir):
        if "tensor" in f.lower():
            print(f"  Found: {f}")