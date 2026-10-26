import re
with open('C:/Users/BRENT/Downloads/handson_fixed/handson/src/inference/labels.ts') as f:
    content = f.read()
labels = re.findall(r'"([^"]+)"', content)
print('LABELS count:', len(labels))
print('First 5:', labels[:5])
print('Last 5:', labels[-5:])