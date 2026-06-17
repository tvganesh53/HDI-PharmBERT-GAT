"""
upload_final.py
Uploads updated code files + Dockerfile to HF Space.
Models are now in tvganesh538/hdi-models — NOT copied into the Space.
"""
from huggingface_hub import HfApi
import time

TOKEN   = "YOUR_SECRET_HERE"
REPO_ID = "tvganesh538/nlp-classifier-api"

api = HfApi(token=TOKEN)

FILES = [
    "predictor_pharmbert.py",
    "predictor_fusion.py",
    "pipeline_hdi.py",
    "app_phase_g.py",
    "Dockerfile",
]

print("=" * 60)
print("Uploading files to HF Space …")
print("=" * 60)

for f in FILES:
    print(f"  Uploading {f} …", end=" ", flush=True)
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id=REPO_ID,
        repo_type="space",
        commit_message=f"Deploy HDI pipeline P8: {f}",
    )
    print("✓")
    time.sleep(1)

# Set HF_TOKEN secret so predictors can download from the model repo
print("\nSetting HF_TOKEN secret …", end=" ", flush=True)
api.add_space_secret(REPO_ID, "HF_TOKEN", TOKEN)
print("✓")

# Set permanent admin key
print("Setting PERMANENT_ADMIN_KEY secret …", end=" ", flush=True)
api.add_space_secret(REPO_ID, "PERMANENT_ADMIN_KEY", "YOUR_SECRET_HERE")
print("✓")

print("\n" + "=" * 60)
print("All done! Space rebuild triggered.")
print("Wait 15–20 min (models download on first boot).")
print(f"Space: https://tvganesh538-nlp-classifier-api.hf.space")
print("=" * 60)
