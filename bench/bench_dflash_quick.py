#!/usr/bin/env python3
"""Quick DFlash γ=16 bench — single prompt, 256-tok cap, hard timeout.

Designed to A/B test ATLAS_*_KGAMMA gates without burning 7 × 4-prompt × 1024-tok
budget. count100 alone is sufficient to detect a real win because it gives
the highest accept rate (drafter is best at predictable token streams).
"""
import sys, time, requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "8890"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "dflash"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 2

URL = f"http://localhost:{PORT}/v1/chat/completions"
try:
    MODEL = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
except Exception as e:
    print(f"[{LABEL}] ERROR: cannot reach :{PORT} — {e}")
    sys.exit(2)

PROMPT = "Count from 1 to 100 separated by commas. Output ONLY the numbers."
tps_list, ct_list = [], []
for r in range(RUNS):
    t0 = time.time()
    try:
        rsp = requests.post(URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 256,
            "temperature": 0.0,
        }, timeout=120)
        rsp.raise_for_status()
        d = rsp.json()
    except Exception as e:
        print(f"[{LABEL}] run {r+1} ERROR ({time.time()-t0:.1f}s): {e}")
        continue
    wall = time.time() - t0
    u = d.get("usage", {})
    ct = u.get("completion_tokens", 0)
    tps = u.get("response_token/s") or (ct/wall if wall > 0 else 0)
    tps_list.append(tps); ct_list.append(ct)

if tps_list:
    mean = sum(tps_list)/len(tps_list)
    peak = max(tps_list)
    print(f"[{LABEL}] runs={len(tps_list)} mean={mean:6.2f} peak={peak:6.2f} ct_avg={sum(ct_list)/len(ct_list):.0f}")
else:
    print(f"[{LABEL}] ALL RUNS FAILED")
