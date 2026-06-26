import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
ADMIN_KEY = 'sk-OvbVd0IEFwQmoSanyZXc_BVTWFP8Hhnf8QasrGPcPr8'

r = httpx.post(f'{BASE}/keys',
    headers={'X-API-Key': ADMIN_KEY},
    json={'name': 'frontend-public', 'role': 'user'},
    timeout=30)
print(r.status_code)
print(json.dumps(r.json(), indent=2))
