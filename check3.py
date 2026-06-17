with open('predictor_pharmbert.py', 'r', encoding='utf-8') as f:
    content = f.read()
idx = content.find('from_pretrained')
print(repr(content[idx-50:idx+300]))
