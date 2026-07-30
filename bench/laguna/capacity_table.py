#!/usr/bin/env python3
"""Render the capacity/throughput table from a capacity_table.sh run.

Three of the four rows are facts the serve prints at boot (weights resident,
KV pool depth) and one is measured (prefill, decode). Everything here is
scraped or read from a JSON written in this sitting -- nothing is carried
over from a previous run, because a stale artifact is exactly how a dead arm
gets scored as a live one.

Decode is quoted SERVER-side (the serve's own `Done: N tokens ... X tok/s`),
not from the client wall clock, and prefill is cross-checked the same way:
the serve prints TTFT per request, so the slope can be re-fit without any
HTTP or client-side term in it at all. If the two fits disagree, the number
is not ready to quote.
"""
import argparse
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import OUT_ROOT  # noqa: E402

DONE = re.compile(r"Done: (\d+) tokens \([a-z]+\) ([\d.]+) tok/s, TTFT=([\d.]+)ms")
# capacity/boot facts, each as (label, regex, group-formatter)
FACTS = [
    ("weights_on_disk_gb", r"([\d.]+) GB on-disk"),
    ("gpu_total_gb", r"GPU 0: ([\d.]+) GB total"),
    ("atlas_resident_gb", r"Atlas-own ([\d.]+) GB"),
    ("cotenant_gb", r"co-tenants ([\d.]+) GB excluded"),
    ("kv_budget_gb", r"([\d.]+) GB budget"),
    ("kv_gb", r"→ ([\d.]+) GB for KV"),
    ("kv_tokens", r"= (\d+) max KV tokens"),
    ("kv_blocks", r"→ (\d+) blocks × \d+ tok/block"),
    ("max_prefill_tokens", r"prefill_budget=(\d+)"),
    ("max_seq_len", r"max_batch_tokens=(\d+)"),
]


def scrape_log(path):
    with open(path, "rb") as f:
        txt = f.read().decode("utf-8", "replace")
    out = {}
    for key, rx in FACTS:
        m = re.search(rx, txt)
        out[key] = float(m.group(1)) if m else None
    out["done"] = [(int(a), float(b), float(c)) for a, b, c in DONE.findall(txt)]
    return out


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def server_side_prefill(done, pre):
    """Re-fit prefill from the serve's OWN TTFT, with no client term.

    prefill_bench issues 1 warm-up + its sweep, all at max_tokens=1, and runs
    before any decode traffic -- so the leading short-completion Done lines are
    exactly its requests. Assert that shape rather than assume it; if the log
    does not match, return None and let the caller say "unavailable" instead of
    fitting whatever happened to be first in the file.
    """
    n = len(pre)
    head = done[:n + 1]
    if len(head) < n + 1 or any(t > 4 for t, _, _ in head):
        return None
    ttft = [c / 1e3 for _, _, c in head[1:]]          # drop the warm-up
    by_len = {}
    for row, t in zip(pre, ttft):
        by_len.setdefault(row["prompt_tokens"] // 64, []).append((row["prompt_tokens"], t))
    xs = [statistics.median(p for p, _ in v) for v in by_len.values()]
    ys = [statistics.median(t for _, t in v) for v in by_len.values()]
    if len(xs) < 3:
        return None
    slope, icept, r2 = fit(xs, ys)
    return {"tok_s": 1 / slope, "intercept_ms": icept * 1e3, "r2": r2}


DECODE_SIZES = {130, 230, 256}          # the decode_bench suite's completion lengths


def server_side_decode(done, pre):
    """Token-weighted decode tok/s from the serve's OWN `Done:` lines, plus a
    contamination check.

    The client wall clock is not usable here and a real run proves it: at
    --max-batch-size 1 a foreign client sharing the serve port queues ahead of
    us, and the client sees that queue wait as slow generation. In one such run
    the serial arm took a burst of NINE foreign 2-16 token requests between
    common-algo and novel-logic; client-side novel-logic read 13.6 tok/s
    (18.8 s) while the serve's own rate for the very same request was
    21.7 tok/s, in line with all five others. Weighted client aggregate 19.2
    vs server 21.8 -- a 12% phantom regression.

    Shape: N one-token prefill probes, then decode_bench's own short warm-up,
    then exactly 6 suite requests. Anything else means someone else was on the
    port; report it and refuse to score, rather than averaging a stranger's
    traffic into ours.
    """
    n_pre = len(pre) + 1                      # sweep + prefill_bench's warm-up
    rest = done[n_pre:]
    if not rest:
        return None, "no decode traffic in the log at all"
    suite = rest[1:]                          # drop decode_bench's warm-up
    foreign = [t for t, _, _ in suite if t not in DECODE_SIZES]
    if foreign or len(suite) != 6:
        return None, (f"CONTAMINATED: expected 6 decode requests of "
                      f"{sorted(DECODE_SIZES)} tokens, saw {len(suite)}"
                      + (f" including foreign completions of {foreign} tokens" if foreign else ""))
    secs = sum(t / r for t, r, _ in suite)
    return sum(t for t, _, _ in suite) / secs, None


def load(d, arm):
    p = os.path.join(d, f"prefill-{arm}.json")
    k = os.path.join(d, f"decode-{arm}.json")
    g = os.path.join(d, f"serve-{arm}.log")
    for f in (p, k, g):
        if not os.path.exists(f):
            sys.exit(f"FATAL: {f} missing -- arm {arm!r} did not complete; refusing "
                     "to render a table with a hole in it")
    return json.load(open(p)), json.load(open(k)), scrape_log(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(OUT_ROOT, "capacity"))
    ap.add_argument("--arms", nargs="*", default=["serial", "dflash"])
    a = ap.parse_args()

    arms = {n: load(a.dir, n) for n in a.arms}
    base = arms[a.arms[0]][2]

    print("=" * 74)
    print("Laguna-S-2.1-NVFP4 on GB10 (1x GB10, 119.7 GB unified LPDDR5x)")
    print("fp8 KV / 8192 ctx / batch 1 / gpu-mem-util 0.80")
    print("=" * 74)

    w = 16
    hdr = "".join(f"{n:>{w}}" for n in a.arms)
    print(f"{'':<30}{hdr}")
    print("-" * (30 + w * len(a.arms)))

    def row(label, fn):
        print(f"{label:<30}" + "".join(f"{fn(arms[n]):>{w}}" for n in a.arms))

    row("weights resident / node", lambda t: f"{t[2]['weights_on_disk_gb']:.1f} GB")
    row("total resident (Atlas)", lambda t: f"{t[2]['atlas_resident_gb']:.1f} GB")
    row("KV pool", lambda t: f"{t[2]['kv_tokens'] / 1000:.0f}K tok")
    srv_dec = {n: server_side_decode(arms[n][2]["done"], arms[n][0]["rows"]) for n in a.arms}

    row("prefill", lambda t: f"{t[0]['prefill_tok_s']:.0f} tok/s")
    print(f"{'decode (weighted, server)':<30}"
          + "".join(f"{(f'{srv_dec[n][0]:.1f} tok/s' if srv_dec[n][0] else 'CONTAMINATED'):>{w}}"
                    for n in a.arms))
    row("  same, client wall clock", lambda t: f"{t[1]['weighted']:.1f} tok/s")
    # An arm with no speculation has no acceptance. Print a dash, never 0.00 --
    # a zero here reads as "speculation ran and accepted nothing".
    def accept(t):
        m = (t[1].get("scrape") or {}).get("mean_accept")
        return f"{m:.2f}" if m else "— (no spec)"
    row("decode accept/step", accept)

    print("-" * (30 + w * len(a.arms)))
    print()
    for n in a.arms:
        if srv_dec[n][1]:
            print(f"!! {n}: {srv_dec[n][1]}")
            print(f"!! {n}: the client-side decode figure above is therefore NOT this arm's "
                  "throughput.\n   The server-side rates are unaffected -- foreign requests were "
                  "served BETWEEN ours,\n   not interleaved into them -- but the client wall clock "
                  "absorbed their queue wait.")
    print("cross-checks (a number nobody can re-derive is not a measurement)")
    for n in a.arms:
        pre, dec, log = arms[n]
        srv = server_side_prefill(log["done"], pre["rows"])
        cli = f"{pre['prefill_tok_s']:.0f} tok/s (R^2 {pre['r2']:.3f})"
        if srv:
            skew = abs(srv["tok_s"] - pre["prefill_tok_s"]) / pre["prefill_tok_s"] * 100
            s = (f"server TTFT fit {srv['tok_s']:.0f} tok/s (R^2 {srv['r2']:.3f}), "
                 f"skew {skew:.1f}%")
            if skew > 10:
                s += "  !! >10% -- do not quote"
        else:
            s = "server TTFT fit UNAVAILABLE (log shape did not match the sweep)"
        print(f"  {n:<8} client fit {cli}; {s}")
        print(f"  {n:<8} fixed per-request overhead {pre['intercept_s'] * 1e3:.0f} ms; "
              f"raw N/latency at N={pre['max_prompt_tokens']} = {pre['raw_tok_s_at_max']:.0f} tok/s")

    print()
    print("KV pool arithmetic, from the serve's own boot log:")
    print(f"  {base['gpu_total_gb']:.1f} GB total x 0.80 util = {base['kv_budget_gb']:.1f} GB budget")
    print(f"  minus {base['atlas_resident_gb']:.1f} GB Atlas-resident and 0.5 GB reserve "
          f"= {base['kv_gb']:.1f} GB for KV")
    print(f"  = {base['kv_blocks']:.0f} blocks x 16 tok = {base['kv_tokens']:.0f} tokens")
    if base["cotenant_gb"]:
        print(f"  !! the serve charges {base['cotenant_gb']:.1f} GB to 'co-tenants' and excludes "
              "it from its own\n     accounting -- but `nvidia-smi --query-compute-apps` lists NO "
              "other process, so on a\n     unified-memory host that figure may be an artifact "
              "of the self-relative heuristic\n     rather than a real neighbour. Either way it is "
              "GB the KV pool does not get.")
    print(f"  per-request ceiling is --max-seq-len, not the pool: "
          f"{base['max_prefill_tokens']:.0f} prefill budget.")
    print()
    print("EP=2 is NOT reproducible on a single-GPU host: only one GPU is visible, so")
    print("there is no second rank to shard experts onto. No EP column is estimated here.")


if __name__ == "__main__":
    main()
