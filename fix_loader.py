from huggingface_hub import hf_hub_download, HfApi
import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
TOKEN   = 'YOUR_SECRET_HERE'
REPO_ID = 'tvganesh538/nlp-classifier-api'

path = hf_hub_download(repo_id=REPO_ID, filename='app_phase_g.py',
                       repo_type='space', token=TOKEN, force_download=True)
with open(path, encoding='utf-8', errors='replace') as f:
    content = f.read()

OLD = '''def _load_predictor() -> Any:
    try:
        from pipeline_hdi import HDIPipeline
        pipe = HDIPipeline()
        pipe.load()
        log.info("HDIPipeline loaded (PharmBERT + PharmFusion)")
        return pipe
    except Exception as exc:
        log.warning("HDIPipeline failed (%s), falling back to PharmBERT only.", exc)
        try:
            from predictor_pharmbert import Predictor
            p = Predictor()
            p.load()
            return p
        except Exception as exc2:
            log.warning("PharmBERT also failed (%s) — stub mode.", exc2)
            return None'''

NEW = '''def _load_predictor() -> Any:
    try:
        from pipeline_hdi import hdi_pipeline
        hdi_pipeline.load()
        log.info("HDIPipeline loaded (PharmBERT P8 + PharmFusion P8)")
        return hdi_pipeline
    except Exception as exc:
        log.warning("HDIPipeline failed (%s), falling back to PharmBERT only.", exc)
        try:
            from predictor_pharmbert import pharmbert_predictor
            pharmbert_predictor.load()
            return pharmbert_predictor
        except Exception as exc2:
            log.warning("PharmBERT also failed (%s) — stub mode.", exc2)
            return None'''

if OLD in content:
    patched = content.replace(OLD, NEW)
    print("Patch applied")
    with open('app_phase_g.py', 'w', encoding='utf-8') as f:
        f.write(patched)
    api = HfApi(token=TOKEN)
    api.upload_file(path_or_fileobj='app_phase_g.py', path_in_repo='app_phase_g.py',
                    repo_id=REPO_ID, repo_type='space',
                    commit_message='Fix _load_predictor to use hdi_pipeline singleton')
    print("Uploaded")
else:
    print("MATCH FAILED - showing current function:")
    idx = content.find('_load_predictor')
    print(repr(content[idx:idx+600]))
