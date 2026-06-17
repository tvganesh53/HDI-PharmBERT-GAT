import torch, os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from huggingface_hub import hf_hub_download
from collections import Counter

path = hf_hub_download(
    repo_id='tvganesh538/hdi-models',
    filename='pharmbert_p8_best.pt',
    cache_dir='./model_cache'
)
ckpt = torch.load(path, map_location='cpu', weights_only=False)

print('label_names:', ckpt.get('label_names'))
print('phase:', ckpt.get('phase'))
print('macro_f1:', ckpt.get('history', {}).get('val_macro_f1'))

# Check bert_labels distribution from the checkpoint
bert_labels = ckpt.get('bert_labels', [])
if bert_labels:
    counts = Counter(bert_labels)
    label_names = ckpt.get('label_names', [])
    print('\nLabel distribution in checkpoint data:')
    total = len(bert_labels)
    for idx in sorted(counts):
        name = label_names[idx] if idx < len(label_names) else str(idx)
        pct = counts[idx]/total*100
        print(f'  {idx} {name:<12}: {counts[idx]:>5} ({pct:.1f}%)')
else:
    print('\nNo bert_labels in checkpoint')
    print('All checkpoint keys:', [k for k in ckpt.keys() if k != 'model_state_dict'])
