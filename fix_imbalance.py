from huggingface_hub import HfApi, hf_hub_download
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

path = hf_hub_download(repo_id=REPO_ID, filename='predictor_pharmbert.py',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8') as f:
    content = f.read()

# Class frequencies from checkpoint: No Effect=2.1%, Possible=5.1%, Positive=8.0%, Negative=7.5%, Harmful=77.3%
# Prior correction: multiply logits by inverse frequency before softmax
OLD = '            with torch.no_grad():\n                logits = self.model(enc["input_ids"], enc["attention_mask"])\n                probs = torch.softmax(logits, dim=-1).squeeze().tolist()'

NEW = '            with torch.no_grad():\n                logits = self.model(enc["input_ids"], enc["attention_mask"])\n                # Prior correction — down-weight Harmful (77.3%) and boost minority classes\n                import torch as _torch\n                freq = _torch.tensor([0.021, 0.051, 0.080, 0.075, 0.773], device=logits.device)\n                correction = _torch.log(1.0 / freq)\n                logits = logits + correction\n                probs = _torch.softmax(logits, dim=-1).squeeze().tolist()'

if OLD in content:
    patched = content.replace(OLD, NEW)
    print('Patch applied')
    with open('predictor_pharmbert.py', 'w', encoding='utf-8') as f:
        f.write(patched)
    api = HfApi(token=TOKEN)
    api.upload_file(path_or_fileobj='predictor_pharmbert.py',
                    path_in_repo='predictor_pharmbert.py',
                    repo_id=REPO_ID, repo_type='space',
                    commit_message='Add prior correction to fix class imbalance bias')
    print('Uploaded')
else:
    print('MATCH FAILED - showing predict block:')
    idx = content.find('with torch.no_grad')
    print(repr(content[idx:idx+300]))
