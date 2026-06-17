from huggingface_hub import hf_hub_download
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
path = hf_hub_download(
    repo_id='tvganesh538/nlp-classifier-api',
    filename='predictor_pharmbert.py',
    repo_type='space',
    token='YOUR_SECRET_HERE',
    force_download=True)
with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()
# Show the load() method - which model file does it load?
idx = content.find('def load')
print(content[idx:idx+800])
