from huggingface_hub import HfApi
api = HfApi(token='YOUR_SECRET_HERE')
api.upload_file(
    path_or_fileobj='pipeline_hdi.py',
    path_in_repo='pipeline_hdi.py',
    repo_id='tvganesh538/nlp-classifier-api',
    repo_type='space',
    commit_message='Fix pipeline_hdi output format to match app_phase_g classify route'
)
print('Uploaded. Waiting for restart...')
