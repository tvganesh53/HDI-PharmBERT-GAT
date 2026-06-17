import httpx, json
BASE = 'https://tvganesh538-nlp-classifier-api.hf.space'
KEY = 'YOUR_SECRET_HERE'

tests = [
    ('[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness', 'Harmful'),
    ('[HERB] Green Tea [DRUG] Ibuprofen [TYPE] Food [REL] no significant interaction', 'No Effect'),
    ('[HERB] Turmeric [DRUG] Metformin [TYPE] Herb [REL] may improve insulin sensitivity', 'Positive'),
    ('[HERB] Ginger [DRUG] Aspirin [TYPE] Herb [REL] possible mild interaction', 'Possible'),
    ('[HERB] Pomegranate [DRUG] Atorvastatin [TYPE] Food [REL] slight CYP3A4 inhibition', 'Negative'),
]

for text, expected in tests:
    r = httpx.post(f'{BASE}/classify', headers={'X-API-Key': KEY},
                   json={'inputs': text}, timeout=120)
    res = r.json()['results'][0]
    got = res.get('top_label', '?')
    match = '✅' if got == expected else '❌'
    print(f'{match} Expected: {expected:<12} Got: {got:<12} ({res.get("top_score",0)*100:.1f}%)')
