#!/usr/bin/env python3
"""Greedy decode probe for one server lifetime.

usage: client.py <port> <prompts.json> <reps> <out.jsonl>

Each prompt gets one discarded warmup then <reps> measured requests. Decode tok/s
excludes prefill: (completion_tokens - 1) / (wall - ttft). Every trial records the
sha256 of reasoning+content so self-determinism and cross-arm identity are both
checkable per trial, not just per arm.
"""
import hashlib
import json
import sys
import time
import urllib.request

port, prompts_path, reps, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
prompts = json.load(open(prompts_path))


def one(p):
    body = {
        "model": "m",
        "temperature": 0,
        "max_tokens": p["max_tokens"],
        "messages": [{"role": "user", "content": p["content"]}],
        "chat_template_kwargs": {"enable_thinking": p["think"]},
        "ignore_eos": False,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    r = json.load(urllib.request.urlopen(req, timeout=900))
    wall_ms = (time.monotonic() - t0) * 1000
    if "error" in r:
        raise RuntimeError(r["error"])
    msg = r["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    u = r["usage"]
    ct = u["completion_tokens"]
    ttft = float(u.get("time_to_first_token_ms") or 0.0)
    dec = (ct - 1) / ((wall_ms - ttft) / 1000) if wall_ms > ttft and ct > 1 else float("nan")
    return {
        "completion_tokens": ct,
        "prompt_tokens": u["prompt_tokens"],
        "finish": r["choices"][0]["finish_reason"],
        "wall_ms": round(wall_ms, 1),
        "ttft_ms": round(ttft, 1),
        "decode_tok_s": round(dec, 3),
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
        "text": text,
    }


with open(out_path, "w") as out:
    for p in prompts:
        for rep in ["warmup"] + list(range(1, reps + 1)):
            rec = one(p)
            rec.update(prompt=p["id"], rep=rep)
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(f"{p['id']:>10} {rep!s:>6} ct={rec['completion_tokens']:4d} "
                  f"dec={rec['decode_tok_s']:7.2f} ttft={rec['ttft_ms']:8.1f} sha={rec['sha']}",
                  flush=True)
