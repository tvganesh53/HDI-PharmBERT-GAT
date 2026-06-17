import httpx
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY  = 'YOUR_API_KEY'
r = httpx.get(f'{BASE}/debug/predictor', headers={'X-API-Key': KEY}, timeout=30)
import json; print(json.dumps(r.json(), indent=2))
