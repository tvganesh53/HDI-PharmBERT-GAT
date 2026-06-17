import torch

ckpt = torch.load('pharmbert_p7_best.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']

# Check if BERT layers have non-trivial weights vs original BioBERT
# Compare first and last encoder layer norms
layer0_w = sd['bert.encoder.layer.0.attention.self.query.weight']
layer11_w = sd['bert.encoder.layer.11.attention.self.query.weight']
classifier_w = sd['classifier.weight']

print('Layer 0 query weight sum:', layer0_w.sum().item())
print('Layer 11 query weight sum:', layer11_w.sum().item())
print('Classifier weight sum:', classifier_w.sum().item())
print('Classifier weight:', classifier_w)
print()
print('Config from checkpoint:', ckpt.get('config'))
print('History:', ckpt.get('history'))
