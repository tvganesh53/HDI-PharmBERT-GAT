import httpx, json

KEY = 'YOUR_API_KEY'

tests = [
    '[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness',
    '[HERB] Garlic [DRUG] Aspirin [TYPE] Herb [REL] affects blood thinners',
    '[HERB] Pomegranate [DRUG] Simvastatin [TYPE] Food [REL] slight effect',
    '[HERB] Ginkgo Biloba [DRUG] Warfarin [TYPE] Herb [REL] increases bleeding risk',
    '[HERB] Green Tea [DRUG] Ibuprofen [TYPE] Food [REL] no significant interaction',
]

for t in tests:
    r = httpx.post(
        'https://tvganesh538-nlp-classifier-api.hf.space/classify',
        headers={'X-API-Key': KEY },
        json={'inputs': t},
        timeout=60
    )
    d = json.loads(r.text)
    if 'results' in d:
        res = d['results'][0]
        print(f'{t[:50]:50s} -> {res["top_label"]} ({res["top_score"]})')
    else:
        print('Error:', d)
