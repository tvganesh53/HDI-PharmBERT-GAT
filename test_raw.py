import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'

r = httpx.get(f'{BASE}/setup', timeout=30)
KEY = r.json()['admin_key']
print('Key:', KEY[:25])

# Get raw text - not json()
r2 = httpx.post(f'{BASE}/classify',
    headers={'X-API-Key': KEY},
    json={'inputs': '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness'},
    timeout=120)
print('Status:', r2.status_code)
print('Headers:', dict(r2.headers))
print('Raw body:', repr(r2.text[:3000]))
