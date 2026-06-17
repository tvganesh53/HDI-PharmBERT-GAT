import os, glob, re

patterns = [
    r'hf_[A-Za-z0-9]{30,}',
    r'gsk_[A-Za-z0-9]{30,}',
    r'sk-[A-Za-z0-9_\-]{20,}',
]

for f in glob.glob('**/*', recursive=True):
    if not os.path.isfile(f):
        continue
    try:
        txt = open(f, encoding='utf-8', errors='ignore').read()
        for p in patterns:
            hits = re.findall(p, txt)
            for h in hits:
                print(f'{f}: {h[:40]}')
    except:
        pass
