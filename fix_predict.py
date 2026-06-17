with open('app_phase_g.py', 'r', encoding='utf-8') as f:
    content = f.read()

old = 'raw = predictor.predict({"inputs": texts})'
new = 'raw = predictor.predict(texts)'

if old in content:
    content = content.replace(old, new)
    with open('app_phase_g.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print('Fixed')
else:
    print('Not found')
