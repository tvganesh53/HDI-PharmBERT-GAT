import httpx
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
r1 = httpx.get(f'{BASE}/', timeout=30)
print('Root:', r1.status_code, r1.text[:200])
r2 = httpx.get(f'{BASE}/health', timeout=30)
print('Health:', r2.status_code, r2.text[:200])
