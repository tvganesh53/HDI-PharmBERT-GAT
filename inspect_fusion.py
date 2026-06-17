import torch
ckpt = torch.load('pharmfusion_p7_best.pt', map_location='cpu', weights_only=False)
print('=== PharmFusion Keys ===')
for k, v in ckpt.items():
    if k != 'model_state_dict':
        print(k, '->', v)
print()
print('=== State Dict Keys ===')
for k, v in ckpt['model_state_dict'].items():
    print(k, '->', v.shape)
