from huggingface_hub import HfApi
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
api = HfApi(token='YOUR_SECRET_HERE')
info = api.space_info(repo_id='tvganesh538/nlp-classifier-api')
for f in sorted(info.siblings, key=lambda x: x.rfilename):
    print(f.rfilename)
