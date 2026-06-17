import httpx, json

texts = [
    'St. Johns Wort reduces effectiveness of warfarin',
    'garlic supplements interact with blood thinners',
    'ginkgo biloba affects aspirin metabolism'
]

for t in texts:
    r = httpx.post(
        'https://tvganesh538-nlp-classifier-api.hf.space/classify',
        headers={'X-API-Key': 'YOUR_SECRET_HERE'},
        json={'inputs': t},
        timeout=60
    )
    d = json.loads(r.text)
    label = d['results'][0]['top_label']
    score = d['results'][0]['top_score']
    print(f'{t[:45]} -> {label} ({score})')
