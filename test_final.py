"""
test_final.py — Full HDI pipeline smoke test (PharmBERT P8 + PharmFusion P8)
Run after the Space finishes rebuilding.
"""
import httpx, json

BASE = "https://tvganesh538-nlp-classifier-api.hf.space"
KEY  = "YOUR_SECRET_HERE"

TESTS = [
    ("[HERB] St Johns Wort [DRUG] Warfarin [TYPE] Herb [REL] reduces effectiveness of anticoagulation",
     "Herb", "Harmful"),
    ("[HERB] Pomegranate [DRUG] Simvastatin [TYPE] Food [REL] slight CYP3A4 effect",
     "Food", "Possible"),
    ("[HERB] Ginkgo Biloba [DRUG] Warfarin [TYPE] Herb [REL] increases bleeding risk significantly",
     "Herb", "Harmful"),
    ("[HERB] Green Tea [DRUG] Ibuprofen [TYPE] Food [REL] no significant interaction observed",
     "Food", "No Effect"),
    ("[HERB] Garlic [DRUG] Aspirin [TYPE] Herb [REL] increases bleeding risk",
     "Herb", "Harmful"),
]

def get_key():
    r = httpx.get(f"{BASE}/setup", timeout=30)
    data = r.json()
    return data.get("admin_key", KEY)

def run_tests(api_key):
    print("=" * 70)
    print("HDI Pipeline Test  —  PharmBERT P8 (severity) + PharmFusion P8 (type)")
    print("=" * 70)

    passed = 0
    for text, exp_type, exp_severity in TESTS:
        r = httpx.post(
            f"{BASE}/classify",
            headers={"X-API-Key": api_key},
            json={"inputs": text},
            timeout=120,
        )
        if r.status_code != 200:
            print(f"  ✗ HTTP {r.status_code}: {r.text[:120]}")
            continue

        d   = r.json()
        res = d["results"][0]
        sev   = res.get("top_label", "?")
        itype = (res.get("interaction_type") or {}).get("top_label", "?")
        summary = res.get("summary", "n/a")

        ok_type = (itype == exp_type)
        ok_sev  = (sev  == exp_severity)
        status  = "✓" if (ok_type and ok_sev) else "✗"
        if ok_type and ok_sev:
            passed += 1

        print(f"  {status} Severity: {sev:<12} Type: {itype:<6}  | {summary}")
        print(f"      Text: {text[:65]}")
        if not ok_type:
            print(f"      ⚠ Expected type={exp_type}, got {itype}")
        if not ok_sev:
            print(f"      ⚠ Expected severity={exp_severity}, got {sev}")
        print()

    print("=" * 70)
    print(f"Result: {passed}/{len(TESTS)} passed")
    print("=" * 70)

if __name__ == "__main__":
    # Try permanent key first; regenerate if 401
    print(f"Testing with key: {KEY[:20]}…")
    r = httpx.get(f"{BASE}/health", timeout=30)
    print(f"Health: {r.json()}")

    # Check predictor status
    r2 = httpx.get(f"{BASE}/debug/predictor",
                   headers={"X-API-Key": KEY}, timeout=30)
    if r2.status_code == 401:
        print("Key expired — regenerating …")
        KEY2 = get_key()
        print(f"New key: {KEY2[:25]}…")
    else:
        KEY2 = KEY
        print(f"Predictor: {json.dumps(r2.json(), indent=2)}")

    run_tests(KEY2)
