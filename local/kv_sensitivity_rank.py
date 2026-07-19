#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""
kv_sensitivity_rank.py — MEASURED per-layer KV-cache precision sensitivity ordering
for the Atlas dense-27B (AEON-Q36) hybrid model on GB10 / RTX 6000.

WHY
---
`--kv-high-precision-layers N` protects the first-N + last-N attention layers at
BF16 KV precision (positional heuristic). But KV-quantization sensitivity is NOT
uniformly positional (buun-llama-cpp / VBR finding): some MIDDLE attention layers
degrade long-context coherence far more than the boundary layers the positional
heuristic happens to protect. Spending the SAME BF16 budget on the layers a sweep
finds most sensitive gives better long-context coherence at identical memory.

This tool produces that ordering by MEASUREMENT (not a weight-only proxy — the KV
cache quantizes runtime K/V ACTIVATIONS, not the on-disk weights, so a weight-stat
proxy measures the wrong thing). It sweeps the model's full-attention layers one at
a time against a served endpoint and ranks them by their measured contribution to
long-context quality.

MODEL FACTS (AEON-Q36-27B config.json)
--------------------------------------
64 layers = 48 linear_attention (SSM, NO paged KV) + 16 full_attention. Only the 16
full-attention layers have a quantizable paged KV cache. Their GLOBAL model layer ids
are [3,7,11,...,63] (full_attention_interval=4), but Atlas indexes the KV-dtype vector
by ATTENTION-LAYER-LOCAL index 0..15 (attention order). This tool ranks + emits the
LOCAL indices, ready to paste into --kv-high-precision-layer-set.

METHOD (leave-one-out protection, KLD-scored)
---------------------------------------------
1. REFERENCE: serve with all 16 attn layers at BF16 KV (the gold distribution).
   Recipe knob:  --kv-cache-dtype bf16   (or --kv-high-precision-layer-set 0,1,...,15)
2. BASELINE:  serve with the target quantized dtype, NO protection
   (--kv-cache-dtype fp8  and empty layer-set). Measure gap vs reference.
3. For each attention layer i in 0..15: serve quantized but with ONLY layer i held
   at BF16 (--kv-high-precision-layer-set i). Measure how much protecting i alone
   recovers toward the reference. The recovery = layer i's SENSITIVITY.
4. RANK layers by recovery, descending. The top-N are the ones to spend the BF16
   budget on:  --kv-high-precision-layer-set <top-N joined by commas>.

Because each config change requires a serve RESTART on GB10, the tool runs in two
modes:
  * --emit-plan   : print the exact serve recipes + curl probes for all 18 configs
                    (reference + baseline + 16 leave-one-out) so an operator (or an
                    orchestration agent) can execute the restart-measure loop. No
                    endpoint needed. This is the default and the "one evening" path.
  * --score DIR   : after the operator has captured each config's probe logprobs to
                    DIR/<config>.jsonl, compute the KLD ranking and emit the ordered
                    --kv-high-precision-layer-set line.

SCORING METRIC
--------------
For each probe token position we compare the quantized config's next-token
distribution to the reference (BF16) distribution via symmetric KL (Jensen-Shannon
is also supported with --metric js). Per-layer sensitivity = mean over probe tokens
of [ KLD(baseline vs ref) - KLD(protect_i vs ref) ]  (how much protecting i closes
the gap). Falls back to a coherence proxy (perplexity of a fixed continuation) when
the endpoint doesn't expose logprobs.

Stdlib only. Read-only against the model files; never restarts a serve itself.
"""
import argparse
import json
import math
import os
import sys
import urllib.request

# ---- model geometry (from AEON-Q36-27B config.json) ---------------------------
FULL_ATTENTION_INTERVAL = 4
NUM_LAYERS = 64
# global model layer ids of the full-attention layers, in order
GLOBAL_ATTN_LAYERS = [i for i in range(NUM_LAYERS) if (i + 1) % FULL_ATTENTION_INTERVAL == 0]
NUM_ATTN = len(GLOBAL_ATTN_LAYERS)  # 16


def _local_to_global(local_idx: int) -> int:
    return GLOBAL_ATTN_LAYERS[local_idx]


# ---- config-plan generation ---------------------------------------------------
def build_configs():
    """The 18 serve configs of the leave-one-out sweep.

    Returns a list of (name, layer_set_arg, kv_dtype_arg) tuples. `layer_set_arg` is
    the value for --kv-high-precision-layer-set ('' = none). For the reference we use
    the full set (all layers BF16) rather than --kv-cache-dtype bf16 so the ONLY thing
    that varies across the sweep is the layer-set — same binary, same everything else.
    """
    configs = []
    all_layers = ",".join(str(i) for i in range(NUM_ATTN))
    configs.append(("reference_bf16all", all_layers, "fp8"))   # every attn layer BF16
    configs.append(("baseline_noprotect", "", "fp8"))          # no protection
    for i in range(NUM_ATTN):
        configs.append((f"protect_local{i:02d}_global{_local_to_global(i)}", str(i), "fp8"))
    return configs


def emit_plan(kv_dtype, serve_script, endpoint, probes_path, out_dir):
    configs = build_configs()
    print(f"# KV per-layer sensitivity sweep — {NUM_ATTN} full-attention layers")
    print(f"# global attn layer ids: {GLOBAL_ATTN_LAYERS}")
    print(f"# quantized dtype under test: {kv_dtype}")
    print(f"# {len(configs)} serve configs (reference + baseline + {NUM_ATTN} leave-one-out)")
    print(f"# probes: {probes_path}   captures -> {out_dir}/<config>.jsonl")
    print("#")
    print("# For each config: (1) restart the serve with the shown layer-set, (2) run the")
    print("#   probe capture, (3) move to the next. Then: kv_sensitivity_rank.py --score", out_dir)
    print("#")
    print("# Clean-slate restart (GB10 unified memory — poll /health, never memory.used):")
    print('#   pkill -9 -f "release/spark serve"')
    print("#   until [ -z \"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)\" ]; do sleep 2; done")
    print(f"#   KV_HP_SET=<set> setsid bash {serve_script} </dev/null > /tmp/kvsweep.log 2>&1 & disown")
    print(f"#   until curl -s {endpoint}/health | grep -q ready; do sleep 3; done")
    print()
    for name, layer_set, dt in configs:
        set_desc = layer_set if layer_set else "(none — pure quantized)"
        print(f"## {name}")
        print(f"#   --kv-cache-dtype {dt} --kv-high-precision-layer-set '{layer_set}'   set={set_desc}")
        print(f"KV_HP_SET='{layer_set}' KV_DTYPE={dt} bash {serve_script}   # then:")
        print(f"python3 {os.path.basename(__file__)} --capture {endpoint} "
              f"--probes {probes_path} --out {os.path.join(out_dir, name + '.jsonl')}")
        print()
    print("# After all captures:")
    print(f"python3 {os.path.basename(__file__)} --score {out_dir} --top 5")


# ---- probe capture (needs a live endpoint) ------------------------------------
def _completions_logprobs(endpoint, prompt, max_tokens, top_logprobs=20, timeout=120):
    """Capture per-position next-token logprob distributions via /v1/completions.

    Returns a list of dicts {token, top: {tok: logprob}} per generated position, or
    None if the endpoint doesn't support logprobs (we then fall back to the coherence
    proxy in --capture)."""
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "logprobs": top_logprobs,
        "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v1/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    ch = (d.get("choices") or [{}])[0]
    lp = ch.get("logprobs") or {}
    toks = lp.get("tokens")
    top = lp.get("top_logprobs")
    if not toks or not top:
        # endpoint returned text but no logprobs → caller falls back to proxy
        return {"text": ch.get("text", ""), "no_logprobs": True}
    positions = []
    for t, tl in zip(toks, top):
        positions.append({"token": t, "top": tl})
    return {"text": ch.get("text", ""), "positions": positions}


def capture(endpoint, probes_path, out_path, max_tokens):
    with open(probes_path) as f:
        probes = [json.loads(line) for line in f if line.strip()]
    rows = []
    for p in probes:
        prompt = p["prompt"]
        res = _completions_logprobs(endpoint, prompt, max_tokens)
        rows.append({"id": p.get("id", "?"), **res})
        tag = "err" if res.get("error") else ("no-logprobs" if res.get("no_logprobs") else "ok")
        print(f"  probe {p.get('id','?'):20} -> {tag}", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"captured {len(rows)} probes -> {out_path}", file=sys.stderr)


# ---- scoring ------------------------------------------------------------------
def _dist_from_top(top):
    """Turn a {token: logprob} top-k dict into a normalized prob dict (renormalized
    over the captured top-k — an approximation, but consistent across configs since
    the same top-k support is compared)."""
    if not top:
        return {}
    m = max(top.values())
    exps = {t: math.exp(lp - m) for t, lp in top.items()}
    z = sum(exps.values()) or 1.0
    return {t: v / z for t, v in exps.items()}


def _kld(p, q):
    """KL(p || q) over the shared support (tokens present in p); q-mass floored."""
    eps = 1e-9
    s = 0.0
    for t, pv in p.items():
        if pv <= 0:
            continue
        qv = q.get(t, eps)
        s += pv * math.log(pv / max(qv, eps))
    return s


def _js(p, q):
    m = {t: 0.5 * (p.get(t, 0.0) + q.get(t, 0.0)) for t in set(p) | set(q)}
    return 0.5 * _kld(p, m) + 0.5 * _kld(q, m)


def _sym_kld(p, q):
    return 0.5 * (_kld(p, q) + _kld(q, p))


def _config_dists(path):
    """Load a captured config: return {probe_id: [pos_dist, ...]} using the top dict
    per position. Skips probes with errors / no logprobs."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") or r.get("no_logprobs") or "positions" not in r:
                continue
            out[r["id"]] = [_dist_from_top(pos.get("top", {})) for pos in r["positions"]]
    return out


def _mean_gap(cfg, ref, metric):
    """Mean per-token divergence of `cfg` from `ref` over shared probes/positions."""
    fn = {"kld": _sym_kld, "js": _js}[metric]
    vals = []
    for pid, ref_positions in ref.items():
        cfg_positions = cfg.get(pid)
        if not cfg_positions:
            continue
        for a, b in zip(cfg_positions, ref_positions):
            if a and b:
                vals.append(fn(a, b))
    return (sum(vals) / len(vals)) if vals else None


def score(out_dir, metric, top_n):
    ref = _config_dists(os.path.join(out_dir, "reference_bf16all.jsonl"))
    base = _config_dists(os.path.join(out_dir, "baseline_noprotect.jsonl"))
    if not ref:
        sys.exit(f"no reference capture in {out_dir} (reference_bf16all.jsonl missing/empty). "
                 "Endpoint may not expose logprobs — see the coherence-proxy note in --help.")
    base_gap = _mean_gap(base, ref, metric)
    if base_gap is None:
        sys.exit("baseline capture has no comparable positions vs reference.")
    print(f"# metric={metric}   baseline gap (no protection) vs BF16 reference = {base_gap:.5f}")
    print(f"# per-layer sensitivity = gap_recovered when that layer alone is protected")
    print()
    rows = []
    for i in range(NUM_ATTN):
        name = f"protect_local{i:02d}_global{_local_to_global(i)}"
        cfg = _config_dists(os.path.join(out_dir, name + ".jsonl"))
        gap = _mean_gap(cfg, ref, metric)
        if gap is None:
            print(f"  local {i:2d} (global {_local_to_global(i):2d}): MISSING capture — skipped", file=sys.stderr)
            continue
        recovered = base_gap - gap  # higher = protecting this layer helps more = more sensitive
        rows.append((i, _local_to_global(i), gap, recovered))
    rows.sort(key=lambda r: r[3], reverse=True)
    print(f"{'rank':>4} {'local':>5} {'global':>6} {'gap_vs_ref':>11} {'recovered':>10}")
    for rank, (li, gi, gap, rec) in enumerate(rows):
        print(f"{rank:>4} {li:>5} {gi:>6} {gap:>11.5f} {rec:>+10.5f}")
    top_locals = [r[0] for r in rows[:top_n]]
    print()
    print(f"# TOP-{top_n} most KV-sensitive attention layers (measured):")
    print(f"--kv-high-precision-layer-set {','.join(str(i) for i in sorted(top_locals))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit-plan", action="store_true",
                    help="print the 18 serve recipes + probes for the restart-measure loop (default)")
    ap.add_argument("--capture", metavar="ENDPOINT",
                    help="capture probe logprobs from a LIVE endpoint into --out (run once per config)")
    ap.add_argument("--score", metavar="DIR",
                    help="score captured configs in DIR and emit the ranked layer-set")
    ap.add_argument("--probes", default=os.path.join(os.path.dirname(__file__), "kv_probes.jsonl"),
                    help="long-context probe prompts (jsonl: {id, prompt})")
    ap.add_argument("--out", help="capture output path (with --capture)")
    ap.add_argument("--out-dir", default="/tmp/kv_sweep", help="capture dir (for --emit-plan)")
    ap.add_argument("--endpoint", default="http://localhost:8890", help="endpoint (for --emit-plan)")
    ap.add_argument("--serve-script", default="local/serve-aeon-27b-dflash.sh",
                    help="serve script the plan restarts with each config")
    ap.add_argument("--kv-dtype", default="fp8", help="quantized dtype under test")
    ap.add_argument("--max-tokens", type=int, default=64, help="probe generation length")
    ap.add_argument("--metric", choices=["kld", "js"], default="kld")
    ap.add_argument("--top", type=int, default=5, help="how many layers the emitted set protects")
    args = ap.parse_args()

    if args.capture:
        if not args.out:
            sys.exit("--capture requires --out <path>")
        capture(args.capture, args.probes, args.out, args.max_tokens)
    elif args.score:
        score(args.score, args.metric, args.top)
    else:
        emit_plan(args.kv_dtype, args.serve_script, args.endpoint, args.probes, args.out_dir)


if __name__ == "__main__":
    main()
