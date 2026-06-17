import httpx, json
r = httpx.post(
    'https://tvganesh538-nlp-classifier-api.hf.space/classify',
    headers={'X-API-Key': 'YOUR_API_KEY'},
    json={'inputs': '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb'},
    timeout=60
)
print(r.status_code, r.text)
