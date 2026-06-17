from huggingface_hub import hf_hub_download
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
path = hf_hub_download(
    repo_id='tvganesh538/nlp-classifier-api',
    filename='app_phase_g.py',
    repo_type='space',
    token='YOUR_SECRET_HERE',
    force_download=True
)
with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()

# Print lifespan function
idx = content.find('lifespan')
print('=== LIFESPAN ===')
print(content[idx:idx+2000])
print()
# Print key_store / api_keys imports
idx2 = content.find('key_store')
print('=== KEY_STORE USAGE ===')
print(content[max(0,idx2-100):idx2+500])
