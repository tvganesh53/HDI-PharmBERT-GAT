with open('predictor_pharmbert.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'MODEL_PATH     = os.getenv("PHARMBERT_MODEL_PATH", "pharmbert_p7_best.pt")',
    'MODEL_PATH     = os.getenv("PHARMBERT_MODEL_PATH", "pharmbert_p8_best.pt")'
)
content = content.replace(
    'LABEL_NAMES    = ["Food", "Herb"]',
    'LABEL_NAMES    = ["No Effect", "Possible", "Positive", "Negative", "Harmful"]'
)

with open('predictor_pharmbert.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Updated predictor_pharmbert.py')
