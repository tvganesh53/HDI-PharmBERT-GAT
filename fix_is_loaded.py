from huggingface_hub import hf_hub_download, HfApi
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

path = hf_hub_download(repo_id=REPO_ID, filename='predictor_pharmbert.py',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()

OLD = '        self._loaded = False\n'
NEW = '        self._loaded = False\n\n    @property\n    def is_loaded(self):\n        return self._loaded\n'

if OLD in content:
    patched = content.replace(OLD, NEW, 1)
    print('Patch applied')
    with open('predictor_pharmbert.py', 'w', encoding='utf-8') as f:
        f.write(patched)
    api = HfApi(token=TOKEN)
    api.upload_file(path_or_fileobj='predictor_pharmbert.py',
                    path_in_repo='predictor_pharmbert.py',
                    repo_id=REPO_ID, repo_type='space',
                    commit_message='Add is_loaded property to PharmBERTPredictor')
    print('Uploaded')
else:
    print('MATCH FAILED - showing init block:')
    idx = content.find('class PharmBERTPredictor')
    print(repr(content[idx:idx+300]))
