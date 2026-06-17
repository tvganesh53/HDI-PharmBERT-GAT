from huggingface_hub import HfApi
api = HfApi(token='YOUR_SECRET_HERE')
logs = api.get_space_runtime(repo_id='tvganesh538/nlp-classifier-api')
import json
print(json.dumps(logs.raw, indent=2, default=str))
