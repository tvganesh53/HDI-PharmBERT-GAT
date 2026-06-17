import httpx, json

r = httpx.get(
    'https://tvganesh538-nlp-classifier-api.hf.space/debug/predictor',
    timeout=60
)
print(r.text)
