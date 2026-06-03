#!/usr/bin/env python3
"""Security-workload bench — measures decode tok/s on prompts shaped like
the actual reads/writes a security researcher does all day: vuln
analysis, PoC writing, CVE explanations, kernel-patch diffs.

Usage:
  bench_security.py <port> [label] [runs] [max_tokens]

Works against any OpenAI-compatible /v1/chat/completions.
"""
import sys
import time
import json
import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "8889"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "atlas"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 512

URL = f"http://localhost:{PORT}/v1/chat/completions"
try:
    MODEL = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
except Exception as e:
    sys.stderr.write(f"ERROR: cannot reach :{PORT} /v1/models — {e}\n")
    sys.exit(2)

# Prompts modeled on actual security-research workloads:
#   poc_write — write a working exploit in C from a vuln description (heap
#               UAF / kernel n-day work — heaviest code-emission category).
#   cve_explain — explain a CVE the way an L2 SOC analyst would (essay
#                 with technical accuracy, matching report-writing).
#   patch_diff — analyze a kernel patch for security implications (mixed
#                code + prose; closest to live bug-hunting reads).
#   tool_call — emit a structured tool call for an external scanner
#               (json/grammar shape — agentic security tooling).
PROMPTS = [
    ("poc_write",
     "Write a complete proof-of-concept exploit in C for a heap "
     "use-after-free vulnerability in a hypothetical kernel io_uring "
     "submission queue when SQE-resize races with a poll registration. "
     "Include the spray allocator setup, the race window trigger, a "
     "msg_msg cross-cache reclaim primitive, and the final kernel-RIP "
     "hijack. Output ONLY the C code with brief inline comments — no prose."),
    ("cve_explain",
     "Write a detailed technical explanation of CVE-2024-1086 (the "
     "Notselwyn netfilter nf_tables nft_verdict_init double-free in the "
     "Linux kernel) suitable for a vulnerability advisory: root-cause, "
     "exploit primitive, affected versions, KASAN signature, and "
     "exploitation difficulty against modern mitigations (SMEP/SMAP/KPTI/"
     "control-flow guards). 500+ words. Plain paragraphs, no bullet lists."),
    ("patch_diff",
     "Given a Linux kernel patch that changes `copy_from_user(buf, "
     "user_ptr, size)` to `copy_from_user(buf, user_ptr, "
     "min(size, sizeof(buf)))`, analyze whether the original was "
     "exploitable. Cover: was there a TOCTOU window, what primitive does "
     "the fix close, can the unfixed version reach SLUB or stack via "
     "either the read or the next-tier kernel allocator, and is the "
     "kernel.org changelog likely to mark this CVE-worthy. Show your "
     "reasoning in 400+ words."),
    ("tool_call",
     "Emit a JSON tool-call request to invoke a fuzzer named "
     "`syzkaller_run` with these parameters: target=`linux-6.19.11`, "
     "subsystem=`io_uring`, duration_minutes=120, kasan=true, "
     "reproducer=true, corpus_seed=`prior_iou_bundle_oob_repro`. Wrap "
     "the call in standard ChatML <tool_call>...</tool_call> tags with "
     "JSON inside. Output the tags + JSON only, no narration."),
]

print(f"[{LABEL}] model={MODEL}  port={PORT}  runs={RUNS}  max_tokens={MAX_TOKENS}")
print(f"{'prompt':<13}  {'mean':>7}  {'peak':>7}  {'ttft':>7}  {'ct':>5}")

results = {}
for name, prompt in PROMPTS:
    tps_list, ttft_list, ct_list = [], [], []
    for _ in range(RUNS):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            # Apples-to-apples with bench_aeon27b.py: disable thinking so
            # decode-only tok/s isn't inflated by reasoning tokens.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        try:
            r = requests.post(URL, json=payload, timeout=240)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"  {name:<13}  ERROR  {e}")
            continue
        wall = time.time() - t0
        u = d.get("usage", {})
        tps = u.get("response_token/s") or u.get("response_tokens_per_second")
        ttft = u.get("time_to_first_token_ms", 0)
        ct = u.get("completion_tokens", 0)
        if not tps and ct and wall > 0:
            tps = ct / wall
        tps_list.append(tps or 0.0)
        ttft_list.append(ttft or 0.0)
        ct_list.append(ct or 0)
    if not tps_list:
        continue
    mean = sum(tps_list) / len(tps_list)
    peak = max(tps_list)
    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_ct = sum(ct_list) / len(ct_list)
    results[name] = {"mean": mean, "peak": peak, "ttft": avg_ttft, "ct": avg_ct}
    print(f"  {name:<13}  {mean:7.2f}  {peak:7.2f}  {avg_ttft:7.0f}  {avg_ct:5.0f}")

if results:
    means = [v["mean"] for v in results.values()]
    print(f"  {'MEAN':<13}  {sum(means)/len(means):7.2f}")

out_path = f"/tmp/bench_security_{LABEL}.json"
with open(out_path, "w") as f:
    json.dump({"label": LABEL, "port": PORT, "model": MODEL, "results": results}, f, indent=2)
print(f"  -> {out_path}")
