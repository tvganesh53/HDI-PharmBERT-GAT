import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY  = 'YOUR_SECRET_HERE'

tests = [
    '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness',
    '[HERB] Pomegranate [DRUG] Simvastatin [TYPE] Food [REL] slight CYP3A4 effect',
    '[HERB] Ginkgo Biloba [DRUG] Warfarin [TYPE] Herb [REL] increases bleeding risk',
    '[HERB] Green Tea [DRUG] Ibuprofen [TYPE] Food [REL] no significant interaction',
    '[HERB] Garlic [DRUG] Aspirin [TYPE] Herb [REL] increases bleeding risk',
]

print('=' * 70)
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
    itype = res.get('interaction_type')
    print('Severity :', res['top_label'], round(res['top_score']*100,1), '%')
    if itype:
        print('Type     :', itype['top_label'], round(itype['top_score']*100,1), '%')
    print('Summary  :', res.get('summary', 'n/a'))
    print('Text     :', text[:55])
    print('-' * 70)
