"""
deploy_to_hf.py — manually push your app to Hugging Face Spaces.

Usage:
    python deploy_to_hf.py

Requirements:
    pip install huggingface_hub
"""

import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────────────────
HF_TOKEN  = "YOUR_SECRET_HERE"
HF_USER   = "tvganesh538"
SPACE_NAME = "nlp-classifier-api"
REPO_ID   = f"{HF_USER}/{SPACE_NAME}"

# Files to upload — must exist in current directory
FILES = [
    "README.md",
    "Dockerfile",
    "requirements_hf.txt",
    "db_adapter.py",
    "app_phase_g.py",
    "predictor.py",
    "pipeline.py",
    "batcher.py",
    "worker.py",
    "schemas.py",
    "queue.py",
    "auth.py",
    "api_keys.py",
    "rate_limiter.py",
]

# ── Deploy ────────────────────────────────────────────────────────────────────
def main():
    api = HfApi(token=HF_TOKEN)

    print(f"Creating/verifying Space: {REPO_ID}")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
    )
    print(f"Space ready: https://huggingface.co/spaces/{REPO_ID}")

    # Set GROQ_API_KEY as a secret in the Space
    print("Setting GROQ_API_KEY secret...")
    try:
        api.add_space_secret(
            repo_id=REPO_ID,
            key="GROQ_API_KEY",
            value=os.getenv("GROQ_API_KEY", "YOUR_GROQ_KEY"),
        )
        api.add_space_secret(
            repo_id=REPO_ID,
            key="DB_BACKEND",
            value="sqlite",
        )
        print("Secrets set.")
    except Exception as e:
        print(f"Secret note: {e}")

    # Upload files
    missing = []
    for f in FILES:
        if Path(f).exists():
            print(f"Uploading {f}...")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=REPO_ID,
                repo_type="space",
            )
        else:
            missing.append(f)
            print(f"  SKIPPED (not found): {f}")

    print("\n" + "="*60)
    print("Deployment complete!")
    print(f"Space URL : https://huggingface.co/spaces/{REPO_ID}")
    print(f"API URL   : https://{HF_USER}-{SPACE_NAME}.hf.space")
    print(f"Docs URL  : https://{HF_USER}-{SPACE_NAME}.hf.space/docs")
    print("="*60)

    if missing:
        print(f"\nWarning: {len(missing)} file(s) not found: {missing}")
        print("Copy them to your project folder and re-run.")

    print("\nNote: Space takes 2-5 minutes to build. Check build logs at:")
    print(f"https://huggingface.co/spaces/{REPO_ID}/logs")

if __name__ == "__main__":
    main()
