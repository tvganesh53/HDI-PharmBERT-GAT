import os, re

patterns = [
    r'hf_[A-Za-z0-9]{30,}',
    r'gsk_[A-Za-z0-9]{30,}',
    r'sk-[A-Za-z0-9_\-]{20,}',
]

skip_dirs = {'venv', '.venv', '__pycache__', '.git', 'node_modules', 'model_cache'}

for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in skip_dirs]
    for f in files:
        if not f.endswith(('.py', '.json', '.yml', '.yaml', '.env', '.txt', '.md')):
            continue
        path = os.path.join(root, f)
        try:
            txt = open(path, encoding='utf-8', errors='ignore').read()
            for p in patterns:
                for hit in re.findall(p, txt):
                    print(f'{path}: {hit[:45]}')
        except:
            pass
