#!/usr/bin/env python3
"""Real tool-call bench — exercises the OpenAI tools=[] parameter that
triggers xgrammar-constrained decode in Atlas. The earlier bench prompts
just asked the model to emit ChatML tool-call tags in free-form; this
one engages the constrained-decode path that production tool callers use.

Usage:
  bench_toolcall_real.py <port> [label] [runs] [max_tokens]
"""
import sys
import time
import json
import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "8889"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "atlas"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 256

URL = f"http://localhost:{PORT}/v1/chat/completions"
try:
    MODEL = requests.get(
        f"http://localhost:{PORT}/v1/models", timeout=5
    ).json()["data"][0]["id"]
except Exception as e:
    sys.stderr.write(f"ERROR: cannot reach :{PORT} /v1/models — {e}\n")
    sys.exit(2)

# Three production-shaped tool schemas covering the three real scenarios
# a security-research agent runs:
#   read_file     — file-IO tool, small JSON args (short response)
#   run_scanner   — multi-arg tool with enum constraint (medium response)
#   write_report  — string-heavy tool with long structured prose (long
#                   response — exercises grammar engine on JSON-quoted text)
SCENARIOS = [
    (
        "read_file",
        "List the relevant files in /home/user/security/findings for the "
        "io_uring io_bundle OOB report. Use the read_file tool to fetch "
        "the most relevant one.",
        [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file from disk and return its contents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_bytes": {"type": "integer", "minimum": 1},
                        },
                        "required": ["path"],
                    },
                },
            }
        ],
    ),
    (
        "run_scanner",
        "Fuzz the Linux io_uring subsystem for KASAN-class memory bugs for "
        "two hours using syzkaller. Resume from the io_bundle_oob corpus.",
        [
            {
                "type": "function",
                "function": {
                    "name": "syzkaller_run",
                    "description": "Run syzkaller against a Linux kernel target.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "subsystem": {
                                "type": "string",
                                "enum": ["io_uring", "netfilter", "fs", "tty", "bpf"],
                            },
                            "duration_minutes": {"type": "integer", "minimum": 1},
                            "kasan": {"type": "boolean"},
                            "reproducer": {"type": "boolean"},
                            "corpus_seed": {"type": "string"},
                        },
                        "required": ["target", "subsystem", "duration_minutes"],
                    },
                },
            }
        ],
    ),
    (
        "write_report",
        "Submit a HackerOne report for CVE-2024-1086 (Notselwyn netfilter "
        "double-free) explaining the root cause, exploit primitive, and "
        "affected versions. Target the H1 program 'kernel'.",
        [
            {
                "type": "function",
                "function": {
                    "name": "h1_submit",
                    "description": "Submit a HackerOne vulnerability report.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "program": {"type": "string"},
                            "title": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["low", "medium", "high", "critical"],
                            },
                            "body": {
                                "type": "string",
                                "description": "Markdown body, 300+ chars.",
                            },
                            "cve_id": {"type": "string"},
                        },
                        "required": ["program", "title", "severity", "body"],
                    },
                },
            }
        ],
    ),
]

print(f"[{LABEL}] model={MODEL}  port={PORT}  runs={RUNS}  max_tokens={MAX_TOKENS}")
print(f"{'scenario':<14}  {'mean':>7}  {'peak':>7}  {'ttft':>7}  {'ct':>5}")

results = {}
for name, prompt, tools in SCENARIOS:
    tps_list, ttft_list, ct_list = [], [], []
    for _ in range(RUNS):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        t0 = time.time()
        try:
            r = requests.post(URL, json=payload, timeout=240)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"  {name:<14}  ERROR  {e}")
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
    print(f"  {name:<14}  {mean:7.2f}  {peak:7.2f}  {avg_ttft:7.0f}  {avg_ct:5.0f}")

if results:
    means = [v["mean"] for v in results.values()]
    print(f"  {'MEAN':<14}  {sum(means)/len(means):7.2f}")

out_path = f"/tmp/bench_toolcall_{LABEL}.json"
with open(out_path, "w") as f:
    json.dump({"label": LABEL, "port": PORT, "model": MODEL, "results": results}, f, indent=2)
print(f"  -> {out_path}")
