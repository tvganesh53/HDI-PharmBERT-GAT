import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY  = 'YOUR_SECRET_HERE'

# Check full debug info
r = httpx.get(f'{BASE}/debug/predictor', headers={'X-API-Key': KEY}, timeout=30)
print('Predictor debug:', json.dumps(r.json(), indent=2))

# Check health for more detail
r2 = httpx.get(f'{BASE}/health', timeout=30)
print('Health:', json.dumps(r2.json(), indent=2))

# Single classify with full response
r3 = httpx.post(f'{BASE}/classify',
    headers={'X-API-Key': KEY},
    json={'inputs': '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness of anticoagulation warfarin bleeding risk'},
    timeout=120)
print('Full response:')
print(json.dumps(r3.json(), indent=2))
