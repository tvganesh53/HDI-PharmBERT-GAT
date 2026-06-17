from huggingface_hub import hf_hub_download, HfApi
import os, re

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN   = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

api  = HfApi(token=TOKEN)
path = hf_hub_download(repo_id=REPO_ID, filename='app_phase_g.py',
                       repo_type='space', token=TOKEN)

with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()

# Show 40 lines around the crash point
idx = content.find('outputs = raw')
print('=== CURRENT CODE (40 chars around crash) ===')
print(content[idx:idx+600])
