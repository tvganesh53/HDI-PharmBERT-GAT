from huggingface_hub import HfApi, hf_hub_download
import hashlib, json, os, time

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN   = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'
PERM_KEY = 'YOUR_SECRET_HERE'

api = HfApi(token=TOKEN)

# Download current keys.json
path = hf_hub_download(repo_id=REPO_ID, filename='keys.json',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8') as f:
    keys = json.load(f)

# Compute hash of permanent key
perm_hash = hashlib.sha256(PERM_KEY.encode()).hexdigest()

# Check if already present
already = any(v.get('key_hash') == perm_hash for v in keys.values())
if already:
    print('Permanent key already in keys.json')
else:
    keys['kid-permanent-admin'] = {
        'key_id':         'kid-permanent-admin',
        'key_hash':       perm_hash,
        'name':           'hf-admin',
        'role':           'admin',
        'created_at':     time.time(),
        'is_active':      True,
        'requests_today': 0,
        'last_used':      None,
    }
    print(f'Added permanent key hash: {perm_hash[:16]}...')

# Keep only essential keys — drop bloated test/fixture keys to slim the file
keep_roles = {'admin'}
slim = {k: v for k, v in keys.items()
        if v.get('name') in ('hf-admin', 'my-admin', 'default-admin')
        or v.get('key_id') == 'kid-permanent-admin'}
slim['kid-permanent-admin'] = keys['kid-permanent-admin']

with open('keys.json', 'w', encoding='utf-8') as f:
    json.dump(slim, f, indent=2)

print(f'keys.json now has {len(slim)} keys')
print('Keys:', [v['name'] for v in slim.values()])

# Upload
api.upload_file(path_or_fileobj='keys.json', path_in_repo='keys.json',
                repo_id=REPO_ID, repo_type='space',
                commit_message='Bake permanent admin key hash into keys.json')
print('Uploaded keys.json')
