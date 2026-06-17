with open('predictor_pharmbert.py', 'rb') as f:
    content = f.read()

idx = content.find(b'results.append')
print(repr(content[idx:idx+300]))
