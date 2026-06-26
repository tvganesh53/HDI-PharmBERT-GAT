import httpx
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
r = httpx.get(f'{BASE}/openapi.json', timeout=30)
import json
data = r.json()
print('Available endpoints:')
for path in data.get('paths', {}).keys():
    print(' ', path)
