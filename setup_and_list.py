import httpx, json

BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'

# Get fresh admin key
r = httpx.get(f'{BASE}/setup', timeout=30)
print('Setup status:', r.status_code)
setup_data = r.json()
print(json.dumps(setup_data, indent=2))
ADMIN_KEY = setup_data['admin_key']

print()

# Now check openapi for available methods/paths
r2 = httpx.get(f'{BASE}/openapi.json', timeout=30)
print('OpenAPI status:', r2.status_code)
data = r2.json()
print('All endpoints with methods:')
for path, methods in data.get('paths', {}).items():
    for method in methods:
        if method in ['get','post','put','delete','patch']:
            print(f'  {method.upper():6s} {path}')
