import httpx, json

tests = [
    'aaaa bbbb cccc',
    '1234 5678',
    'St Johns Wort warfarin interaction herb drug',
    'the quick brown fox jumps over the lazy dog'
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
    print(f'{t[:35]:35s} -> {res["top_label"]} ({res["top_score"]}) all={res["classifications"]}')
