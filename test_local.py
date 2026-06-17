import torch
from transformers import AutoTokenizer, BertForSequenceClassification

ckpt = torch.load('pharmbert_p8_best.pt', map_location='cpu', weights_only=False)
tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
model = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.2', num_labels=5)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
model.classifier.weight.data.copy_(ckpt['model_state_dict']['classifier.weight'])
model.classifier.bias.data.copy_(ckpt['model_state_dict']['classifier.bias'])
model.eval()

labels = ['No Effect', 'Possible', 'Positive', 'Negative', 'Harmful']
tests = [
    '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness',
    '[HERB] Pomegranate [DRUG] Simvastatin [TYPE] Food [REL] no significant effect',
    'random unrelated text about weather',
]

for t in tests:
    enc = tokenizer(t, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=-1)[0]
    print(f'{t[:50]:50s} -> {labels[probs.argmax()]} ({probs.max():.4f})')
    print(f'  All: {[round(p,4) for p in probs.tolist()]}')
