import os, re

hits = []
pat = re.compile(r'(github\.com[^\s"\'<>]+|storage\.googleapis\.com[^\s"\'<>]+|download[^\s"\'<>]*model[^\s"\'<>]*|asl[^\s"\'<>]*\.h5|asl[^\s"\'<>]*\.json|sign[^\s"\'<>]*\.glb|avatar[^\s"\'<>]*\.glb)', re.I)

for root, _, fs in os.walk('C:/Users/BRENT/Downloads/handson_fixed'):
    if 'node_modules' in root or '.venv' in root:
        continue
    for f in fs:
        if f.endswith(('.py', '.ts', '.js', '.md', '.json', '.sh', '.ps1', '.cmd', '.txt')):
            p = os.path.join(root, f)
            try:
                t = open(p, encoding='utf-8', errors='ignore').read()
            except Exception:
                continue
            for m in pat.finditer(t):
                hits.append((os.path.relpath(p, 'C:/Users/BRENT/Downloads/handson_fixed'), m.group(0)))

seen = set()
for p, u in hits:
    if u in seen:
        continue
    seen.add(u)
    print(u, '  <-', p)