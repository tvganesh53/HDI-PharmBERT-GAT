import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'

# Regenerate key
r = httpx.get(f'{BASE}/setup', timeout=30)
data = r.json()
key = data['admin_key']
print('New key:', key)

# Test classify immediately
r2 = httpx.post(f'{BASE}/classify',
    headers={'X-API-Key': key},
    json={'inputs': '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness'},
    timeout=120)
print('Status:', r2.status_code)
print(json.dumps(r2.json(), indent=2))
