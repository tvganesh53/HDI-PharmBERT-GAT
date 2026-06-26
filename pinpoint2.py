import re, os

patterns = [
    r'hf_[A-Za-z0-9]{30,}',
    r'gsk_[A-Za-z0-9]{30,}',
]

skip = {'venv', '.venv', '__pycache__', '.git', 'node_modules', 'model_cache'}

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in skip]
    for fname in files:
        path = os.path.join(root, fname)
        try:
            lines = open(path, encoding='utf-8', errors='ignore').readlines()
            for i, line in enumerate(lines, 1):
                for p in patterns:
                    if re.search(p, line):
                        print(f'{path} line {i}: {line.strip()[:80]}')
        except:
            pass
