from huggingface_hub import hf_hub_download
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN   = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

path = hf_hub_download(repo_id=REPO_ID, filename='app_phase_g.py',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()

idx = content.find('_load_predictor')
print('=== _load_predictor ===')
print(content[idx:idx+1000])
