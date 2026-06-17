with open('predictor_pharmbert.py', 'r', encoding='utf-8') as f:
    content = f.read()
idx = content.find('Build model architecture')
print(content[idx:idx+300])
