import requests
r = requests.get(
    'https://huggingface.co/api/spaces/tvganesh538/nlp-classifier-api/runtime',
    headers={'Authorization': 'Bearer YOUR_SECRET_HERE'}
)
import json
print(json.dumps(r.json(), indent=2))
