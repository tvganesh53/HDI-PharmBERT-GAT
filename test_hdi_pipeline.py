"""
test_hdi_pipeline.py
Tests the full PharmBERT + PharmFusion pipeline on HF Spaces.
Run after deploy_hdi_pipeline.py and HF rebuild completes.
"""
import httpx, json

BASE = "https://tvganesh538-nlp-classifier-api.hf.space"
KEY  = "YOUR_API_KEY"   # update if you regenerated via /setup

tests = [
    # (text,                                              description)
    ("[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness",
     "St Johns Wort + Warfarin — classic herb interaction"),
    ("[HERB] Pomegranate [DRUG] Simvastatin [TYPE] Food [REL] slight CYP3A4 effect",
     "Pomegranate + Simvastatin — food interaction"),
    ("[HERB] Garlic [DRUG] Aspirin [TYPE] Herb [REL] increases bleeding risk",
     "Garlic + Aspirin — antiplatelet risk"),
    ("[HERB] Green Tea [DRUG] Ibuprofen [TYPE] Food [REL] no significant interaction",
     "Green Tea + Ibuprofen — no effect"),
    ("[HERB] Ginkgo Biloba [DRUG] Warfarin [TYPE] Herb [REL] increases bleeding risk significantly",
     "Ginkgo + Warfarin — harmful"),
]

print("=" * 70)
print("Full HDI Pipeline Test (PharmBERT severity + PharmFusion type)")
print("=" * 70)

for text, desc in tests:
    r = httpx.post(
        f"{BASE}/classify",
        headers={"X-API-Key": KEY},
        json={"inputs": text},
        timeout=120,
    )
    if r.status_code != 200:
        print(f"\n✗  {desc}")
        print(f"   Error {r.status_code}: {r.text[:100]}")
        continue

    d = r.json()
    res = d["results"][0]

    # Check if HDIPipeline result format (has severity + interaction_type)
    # or legacy PharmBERT-only format
    raw = res.get("raw_output") or res

    print(f"\n{'─'*70}")
    print(f"📝 {desc}")
    print(f"   Text: {text[:60]}…")
    print(f"   Top label : {res['top_label']}  ({res['top_score']*100:.1f}%)")
    print(f"   Model     : {d.get('model_name', 'unknown')}")
    print(f"   Latency   : {d.get('latency_ms', 0):.0f}ms")

    # Show all class scores
    for cls in res.get("classifications", []):
        bar = "█" * int(cls["score"] * 20)
        print(f"     {cls['label']:12s} {cls['score']:.4f}  {bar}")

print("\n" + "=" * 70)
print("Test complete.")
print("=" * 70)
