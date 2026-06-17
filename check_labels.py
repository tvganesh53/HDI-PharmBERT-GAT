import torch, os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='tvganesh538/hdi-models',
    filename='pharmbert_p8_best.pt',
    cache_dir='./model_cache'
)
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print('label_names:', ckpt.get('label_names'))
print('label2id:',   ckpt.get('label2id'))
print('id2label:',   ckpt.get('id2label'))
print('classes:',    ckpt.get('classes'))
print('config:',     ckpt.get('config'))
# Print ALL top-level keys except state dict
for k, v in ckpt.items():
    if k != 'model_state_dict':
        print(f'{k}: {v}')
