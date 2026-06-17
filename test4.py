import httpx, json

# Test if classify endpoint shows actual PharmBERT scores or stub
r = httpx.get(
    'https://tvganesh538-nlp-classifier-api.hf.space/debug/predictor',
    timeout=60
)
print('Debug:', r.text)

# Check classify with full response
r2 = httpx.post(
    'https://tvganesh538-nlp-classifier-api.hf.space/classify',
    headers={'X-API-Key': 'YOUR_SECRET_HERE'},
    json={'inputs': 'St Johns Wort warfarin'},
    timeout=60
)
print('Classify:', r2.text)
