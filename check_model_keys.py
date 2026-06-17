import torch

ckpt = torch.load('pharmbert_p7_best.pt', map_location='cpu', weights_only=False)
sd = ckpt['model_state_dict']

print('=== All keys ===')
for k in sd.keys():
    print(k, '->', sd[k].shape)
