from huggingface_hub import HfApi, hf_hub_download
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

path = hf_hub_download(repo_id=REPO_ID, filename='predictor_pharmbert.py',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8') as f:
    content = f.read()

OLD = 'LABEL_NAMES = ["Harmful", "Negative", "No Effect", "Positive", "Possible"]'
NEW = 'LABEL_NAMES = ["No Effect", "Possible", "Positive", "Negative", "Harmful"]'

if OLD in content:
    patched = content.replace(OLD, NEW)
    print('Patch applied')
    with open('predictor_pharmbert.py', 'w', encoding='utf-8') as f:
        f.write(patched)
    api = HfApi(token=TOKEN)
    api.upload_file(path_or_fileobj='predictor_pharmbert.py',
                    path_in_repo='predictor_pharmbert.py',
                    repo_id=REPO_ID, repo_type='space',
                    commit_message='Fix LABEL_NAMES order to match checkpoint')
    print('Uploaded')
else:
    print('MATCH FAILED - current line:')
    idx = content.find('LABEL_NAMES')
    print(repr(content[idx:idx+80]))
