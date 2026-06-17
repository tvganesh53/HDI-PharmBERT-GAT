import torch
ckpt = torch.load('pharmgat_p7_best (2).pt', map_location='cpu', weights_only=False)
print('=== GAT Model Keys ===')
for k, v in ckpt.items():
    if k != 'model_state_dict':
        print(k, '->', str(v)[:150])
    else:
        print('model_state_dict keys:')
        for sk, sv in v.items():
            print(' ', sk, '->', sv.shape)
