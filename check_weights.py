import torch

ckpt = torch.load('pharmbert_p7_best.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']

print('Classifier weight:', sd.get('classifier.weight'))
print('Classifier bias:', sd.get('classifier.bias'))
print('All keys with classifier:', [k for k in sd.keys() if 'classifier' in k])
