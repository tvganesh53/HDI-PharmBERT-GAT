from huggingface_hub import hf_hub_download, HfApi
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

path = hf_hub_download(repo_id=REPO_ID, filename='predictor_pharmbert.py',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()

print('=== current predict() return block ===')
idx = content.find('def predict')
print(repr(content[idx:idx+800]))
