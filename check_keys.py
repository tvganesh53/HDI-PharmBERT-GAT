from huggingface_hub import hf_hub_download, HfApi
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN   = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

# Check keys.json on Space
path = hf_hub_download(repo_id=REPO_ID, filename='keys.json',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8', errors='replace') as f:
    print('=== keys.json ===')
    print(f.read())

# Check the patched lifespan block
path2 = hf_hub_download(repo_id=REPO_ID, filename='app_phase_g.py',
                        repo_type='space', token=TOKEN, force_download=True)
with open(path2, encoding='utf-8', errors='replace') as f:
    content = f.read()
idx = content.find('permanent_key = os.getenv')
print()
print('=== LIFESPAN KEY BLOCK ===')
print(content[idx:idx+700])
