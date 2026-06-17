from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id='tvganesh538/nlp-classifier-api',
    filename='app_phase_g.py',
    repo_type='space'
)
with open(path) as f:
    content = f.read()
# Find the classify route
idx = content.find('def classify')
if idx == -1:
    idx = content.find('/classify')
print(content[max(0,idx-200):idx+800])
