import re

patterns = [
    r'hf_[A-Za-z0-9]{30,}',
    r'gsk_[A-Za-z0-9]{30,}',
]

files = [
    'keys_hf.json', 'app_phase_g.py', 'auth.py', 'api_keys.py',
    'deploy_to_hf.py', 'upload_final.py', 'upload_pipeline.py',
    'bake_key.py', 'docker-compose.yml', 'ci.yml', '.github/workflows/ci.yml',
    'DEPLOY_GUIDE.md', 'README.md', 'body.json', 'build.log',
    'logs/classifications.jsonl'
]

for f in files:
    try:
        lines = open(f, encoding='utf-8', errors='ignore').readlines()
        for i, line in enumerate(lines, 1):
            for p in patterns:
                if re.search(p, line):
                    print(f'{f} line {i}: {line.strip()[:80]}')
    except:
        pass
