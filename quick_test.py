import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY  = 'YOUR_API_KEY'

tests = [
    '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness',
    '[HERB] Pomegranate [DRUG] Simvastatin [TYPE] Food [REL] slight CYP3A4 effect',
    '[HERB] Ginkgo Biloba [DRUG] Warfarin [TYPE] Herb [REL] increases bleeding risk significantly',
]
for text in tests:
    r = httpx.post(f'{BASE}/classify',
        headers={'X-API-Key': KEY},
        json={'inputs': text},
        timeout=120)
    d = r.json()
    if r.status_code != 200:
        print('ERROR:', d)
        continue
    res = d['results'][0]
    print('Label:', res['top_label'], ' Score:', res['top_score'], ' Model:', d['model_name'])
    print('  Text:', text[:60])
    print()
