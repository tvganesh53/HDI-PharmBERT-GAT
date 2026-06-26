import httpx
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
r = httpx.get(f'{BASE}/openapi.json', timeout=30)
data = r.json()
for path, methods in data.get('paths', {}).items():
    for method in methods:
        print(method.upper(), path)
