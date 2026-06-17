import os, glob

secrets = [
    'YOUR_SECRET_HERE',
    'YOUR_SECRET_HERE',
    'YOUR_SECRET_HERE',
]

for f in glob.glob('*.py'):
    txt = open(f, encoding='utf-8', errors='ignore').read()
    changed = False
    for s in secrets:
        if s in txt:
            txt = txt.replace(s, 'YOUR_SECRET_HERE')
            print(f'Replaced in {f}: {s[:20]}...')
            changed = True
    if changed:
        open(f, 'w', encoding='utf-8').write(txt)
print('Done')
