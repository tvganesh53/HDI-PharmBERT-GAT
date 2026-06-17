import httpx, json

# Test with very different inputs to see if scores change at all
tests = [
    'warfarin',
    'St Johns Wort herb supplement',
    'this is a completely unrelated sentence about weather',
    'drug interaction serious harmful effect'
]

for t in tests:
    r = httpx.post(
        'https://tvganesh538-nlp-classifier-api.hf.space/classify',
        headers={'X-API-Key': 'YOUR_SECRET_HERE'},
        json={'inputs': t},
        timeout=60
    )
    d = json.loads(r.text)
    res = d['results'][0]
    print(f'{t[:40]:40s} -> {res["top_label"]} ({res["top_score"]})')
