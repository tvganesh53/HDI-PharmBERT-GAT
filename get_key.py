import httpx
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
r = httpx.get(f'{BASE}/setup', timeout=30)
print(r.json())
