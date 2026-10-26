import os

path = r"C:\Users\BRENT\Downloads\handson_fixed\handson\training\model_out\tfjs_model_399.convert.err"
if os.path.exists(path):
    with open(path, 'r') as f:
        print(f.read())
else:
    print(f"NOT FOUND: {path}")
    # List what's in model_out
    print("\nFiles in model_out:")
    for root, dirs, files in os.walk(r"C:\Users\BRENT\Downloads\handson_fixed\handson\training\model_out"):
        for f in files:
            full = os.path.join(root, f)
            print(f"  {full}")