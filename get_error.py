import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY  = 'YOUR_API_KEY'

# Get raw response text to see actual error
r = httpx.post(f'{BASE}/classify',
    headers={'X-API-Key': KEY},
    json={'inputs': '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness'},
    timeout=120)
print('Status:', r.status_code)
print('Raw body:', r.text[:2000])
