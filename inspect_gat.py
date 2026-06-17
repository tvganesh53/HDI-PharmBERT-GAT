import torch
data = torch.load('pharmgat_p7_node_embeddings (2).pt', map_location='cpu', weights_only=False)
print('Type:', type(data))
if isinstance(data, dict):
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(k, '->', v.shape)
        elif isinstance(v, dict):
            print(k, '-> dict with', len(v), 'entries, sample:', list(v.items())[:3])
        else:
            print(k, '->', str(v)[:150])
