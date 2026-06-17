from huggingface_hub import hf_hub_download, HfApi
import os, hashlib

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN   = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

path = hf_hub_download(repo_id=REPO_ID, filename='app_phase_g.py',
                       repo_type='space', token=TOKEN, force_download=True)

with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()

OLD = '''    # Load permanent admin key from HF secret (survives rebuilds)
    permanent_key = os.getenv("PERMANENT_ADMIN_KEY")
    if permanent_key:
        try:
            from api_keys import key_store
            # Try import_raw first, fall back to create if not available
            if hasattr(key_store, "import_raw"):
                key_store.import_raw("hf-admin", "admin", permanent_key)
            else:
                key_store._keys[permanent_key] = type("K", (), {
                    "name": "hf-admin", "role": "admin",
                    "key_id": "hf-admin", "label": "hf-admin"
                })()
            log.info("Permanent admin key loaded from env.")
        except Exception as exc:
            log.warning("Could not load permanent key: %s", exc)'''

NEW = '''    # Load permanent admin key — write hash into keys.json so
    # validate() (which re-reads disk on every request) always finds it.
    permanent_key = os.getenv("PERMANENT_ADMIN_KEY")
    if permanent_key:
        try:
            import hashlib as _hl, time as _time
            from api_keys import key_store, APIKey
            perm_hash = _hl.sha256(permanent_key.encode()).hexdigest()
            already = any(k.key_hash == perm_hash for k in key_store._keys.values())
            if not already:
                perm_obj = APIKey(
                    key_id="kid-permanent-admin",
                    key_hash=perm_hash,
                    name="hf-admin",
                    role="admin",
                    created_at=_time.time(),
                    is_active=True,
                )
                key_store._keys["kid-permanent-admin"] = perm_obj
                key_store._save()
                log.info("Permanent admin key written to keys.json.")
            else:
                log.info("Permanent admin key already present.")
        except Exception as exc:
            log.warning("Could not load permanent key: %s", exc)'''

if OLD in content:
    patched = content.replace(OLD, NEW)
    print("Patch applied")
    with open('app_phase_g.py', 'w', encoding='utf-8') as f:
        f.write(patched)
    api = HfApi(token=TOKEN)
    api.upload_file(path_or_fileobj='app_phase_g.py', path_in_repo='app_phase_g.py',
                    repo_id=REPO_ID, repo_type='space',
                    commit_message='Fix permanent key: write hash to keys.json')
    print("Uploaded")
else:
    print("MATCH FAILED — showing block:")
    idx = content.find('permanent_key = os.getenv')
    print(repr(content[idx:idx+600]))
