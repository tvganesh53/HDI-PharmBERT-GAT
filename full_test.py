import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY  = 'YOUR_API_KEY'

r = httpx.post(f'{BASE}/classify',
    headers={'X-API-Key': KEY},
    json={'inputs': '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness'},
    timeout=120)
print(json.dumps(r.json(), indent=2))
