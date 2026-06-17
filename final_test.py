import httpx, json, time
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY  = 'YOUR_SECRET_HERE'

print('Waiting 60s for Space to pick up new keys.json...')
time.sleep(60)

# Health
r = httpx.get(f'{BASE}/health', timeout=30)
print('Health:', r.json())

# Test with permanent key — no /setup needed
tests = [
    '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness',
    '[HERB] Pomegranate [DRUG] Simvastatin [TYPE] Food [REL] slight CYP3A4 effect',
    '[HERB] Ginkgo Biloba [DRUG] Warfarin [TYPE] Herb [REL] increases bleeding risk',
    '[HERB] Green Tea [DRUG] Ibuprofen [TYPE] Food [REL] no significant interaction',
    '[HERB] Garlic [DRUG] Aspirin [TYPE] Herb [REL] increases bleeding risk',
]

print()
print('=' * 70)
passed = 0
for text in tests:
    r = httpx.post(f'{BASE}/classify',
        headers={'X-API-Key': KEY},
        json={'inputs': text},
        timeout=120)
    if r.status_code != 200:
        print(f'ERROR {r.status_code}: {r.text[:100]}')
        continue
    d   = r.json()
    res = d['results'][0]
    sev   = res.get('top_label', '?')
    itype = (res.get('interaction_type') or {}).get('top_label', '?')
    summary = res.get('summary', 'n/a')
    passed += 1
    print(f'Severity: {sev:<12} Type: {itype:<6} | {summary}')
    print(f'  {text[:65]}')
    print()

print('=' * 70)
print(f'Result: {passed}/{len(tests)} passed')
