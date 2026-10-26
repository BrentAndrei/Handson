import subprocess, sys

# Check if tensorflowjs_converter is available
try:
    result = subprocess.run([sys.executable, "-m", "tensorflowjs_converter", "--help"], 
                          capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print("tensorflowjs_converter available via python -m")
        print(result.stdout[:500])
    else:
        print("NOT available via -m:", result.stderr[:200])
except Exception as e:
    print(f"Error: {e}")

# Check pip list
try:
    result = subprocess.run([sys.executable, "-m", "pip", "show", "tensorflowjs"], 
                          capture_output=True, text=True, timeout=30)
    print(result.stdout[:500])
except Exception as e:
    print(f"pip show error: {e}")