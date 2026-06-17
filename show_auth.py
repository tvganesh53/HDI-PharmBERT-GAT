from huggingface_hub import hf_hub_download
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN = 'YOUR_SECRET_HERE'

for fname in ['api_keys.py', 'auth.py']:
    path = hf_hub_download(
        repo_id='tvganesh538/nlp-classifier-api',
        filename=fname,
        repo_type='space',
        token=TOKEN,
        force_download=True
    )
    with open(path, encoding='utf-8', errors='replace') as f:
        content = f.read()
    print(f'=== {fname} ===')
    print(content)
    print()
