#!/usr/bin/env python3
"""Score the ATLAS_PREFILL_CUBLAS A/B (dense-FFN + head-gate onto cuBLASLt).

THREE ARMS ON PURPOSE: base, cublas, base2.
  * base vs base2 is an A/A pair -- it measures this harness's own floor for
    this session, so the cublas delta is read against a measured floor rather
    than an inherited one. A floor belongs to a harness+config+session.
  * the contamination guard needs a CONSENSUS to self-calibrate, and a
    consensus needs >=3 arms. With 2 it can only report UNCHECKED, which is a
    third state and NOT "clean".

FOUR THINGS THIS REFUSES TO CONFLATE

  [1] DISPATCH. Every arm must print a route line for all three patched sites
      (DENSE_FFN, HEAD_GATE[paged], HEAD_GATE[cache_skip]). A MISSING line is
      not a pass -- it means that layer never executed, so the arm proves
      nothing about it. That is the exact failure that hid a claimed 33% cost
      from the phase table: dense_ffn.rs carries no prof_step, so "absent"
      read as "small". Absent is now its own verdict.

  [2] CONTAMINATION. Request census (multiset of `Done: N tokens`), not any
      content feature. Laguna-S thinks by default via its bundled template, so
      a `Thinking`-keyed guard flags our own traffic -- it did, on 5/5 arms.

  [3] SPEED. Prefill slope, per-request overhead removed by fitting
      latency = a*N + b and taking 1/a.

  [4] CORRECTNESS. cuBLASLt and dense_gemm_tc differ in accumulation order, so
      this is NOT expected to be bit-exact and hash inequality is NOT by itself
      a failure. The absolute needle oracle is what decides, and the base/base2
      hash pair calibrates whether hashes are even stable arm-to-arm here.
"""

import argparse
import json
import os
import re
import sys

NOISE_PCT = 1.6          # launch-to-launch reproducibility, x sqrt(2)
R2_MIN = 0.97
DONE_RE = re.compile(r"Done: (\d+) tokens")
ARRIVE_RE = re.compile(
    r"T(\d\d):(\d\d):(\d\d)\.\d+Z.*Chunked prefill start: (\d+) prompt tokens"
    r", chunk_size=\d+, max_tokens=(\d+)")
SLOPE_MT = 1            # prefill_bench requests one token; that IS the slope
ROUTE_RE = re.compile(r"(DENSE_FFN|HEAD_GATE) prefill(\[[a-z_]+\])? route=(\w+)")

SITES = ("DENSE_FFN", "HEAD_GATE[paged]", "HEAD_GATE[cache_skip]")


def load(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (ValueError, OSError):
        return None


def routes(path):
    """{site: route} from the one-shot dispatch lines. None if no log."""
    if not os.path.exists(path):
        return None
    found = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ROUTE_RE.search(line)
            if m:
                site = m.group(1) + (m.group(2) or "")
                found[site] = m.group(3)
    return found


def census(path):
    """[(t_seconds, prompt_tokens, max_tokens)] for every request ARRIVAL.

    Arrivals, not completions. A completion-based census is blind to a request
    that arrives and never finishes -- base2's intruder landed at the instant
    the serve was killed, produced no `Done` line, and was scored as clean.
    Arrivals also carry the timestamp needed to ask WHEN, which is the whole
    question: the scheduler is max_batch=1 + fifo, so a request that arrives
    AFTER ours queues behind it and cannot touch its latency. Only foreign work
    interleaved INTO the measurement window can bias the slope.
    """
    if not os.path.exists(path):
        return None
    out = []
    with open(path, errors="replace") as fh:
        for line in fh:
            m = ARRIVE_RE.search(line)
            if m:
                h, mi, s = int(m.group(1)), int(m.group(2)), int(m.group(3))
                out.append((h * 3600 + mi * 60 + s, int(m.group(4)), int(m.group(5))))
    return out


def split_foreign(cen):
    """(ours, foreign) per arm, by cross-arm consensus on (prompt_tokens, max_tokens).

    Ours = the request shape every arm received. The sweep is deterministic and
    identical by construction, so anything one arm saw and another did not is
    not ours. Keyed on shape rather than on any content feature: this model
    thinks by default via its bundled template, so a `Thinking`-keyed guard
    flags our own traffic -- it has, on 5/5 arms.
    """
    from collections import Counter
    keys = [Counter((p, m) for _, p, m in c) for c in cen.values()]
    shared = keys[0].copy()
    for k in keys[1:]:
        for key in list(shared):
            shared[key] = min(shared[key], k.get(key, 0))
            if not shared[key]:
                del shared[key]
    ours, foreign = {}, {}
    for t, c in cen.items():
        budget, o, f = shared.copy(), [], []
        for rec in c:
            key = (rec[1], rec[2])
            if budget.get(key, 0) > 0:
                budget[key] -= 1
                o.append(rec)
            else:
                f.append(rec)
        ours[t], foreign[t] = o, f
    return ours, foreign


def slope(p):
    """(tok_s, r2, intercept_ms, max_n) from a prefill_bench.py artifact.

    Read straight out of the artifact rather than re-fitting here. prefill_bench
    fits over the per-length MEDIAN, which is not the same estimator as a fit
    over every raw trial -- and two slopes with the same name that disagree by a
    percent is exactly the sort of thing that gets quoted as a win.
    """
    try:
        return (float(p["prefill_tok_s"]), float(p["r2"]),
                float(p["intercept_s"]) * 1000.0, int(p["max_prompt_tokens"]))
    except (KeyError, TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    a = ap.parse_args()
    d, arms = a.dir, a.arms
    errs = []

    print("=" * 74)
    print("PREFILL cuBLASLt (dense-FFN + head-gate) -- SCORE")
    print("=" * 74)

    # ---------------------------------------------------------------- [1]
    print("\n[1] DISPATCH -- did each patched site actually run, and on which route?")
    rt = {t: routes(f"{d}/serve-{t}.log") for t in arms}
    for t in arms:
        r = rt[t]
        if r is None:
            errs.append(f"{t}: no serve log -- arm did not run")
            print(f"    {t:<8} NO SERVE LOG")
            continue
        cells = []
        for s in SITES:
            v = r.get(s)
            cells.append(f"{s}={v if v else 'ABSENT'}")
        print(f"    {t:<8} " + "  ".join(cells))

    # Absent in EVERY arm and absent in SOME arms are different findings.
    # Symmetric absence = that route does not execute in this config at all, so
    # it is missing from both sides of the subtraction and cannot bias the
    # delta; the honest verdict is that the site is UNTESTED, not that the run
    # is void. Asymmetric absence is fatal -- it means the arms ran different
    # code, which is precisely the confound an A/B exists to exclude.
    live = [t for t in arms if rt[t] is not None]
    for s in SITES:
        saw = [t for t in live if (rt[t] or {}).get(s)]
        if live and not saw:
            print(f"    NOTE: {s} never executed in ANY arm -- this config does "
                  f"not reach that route, so the patch there is UNTESTED "
                  f"(untested is not broken, and it is not working either).")
        elif saw and len(saw) != len(live):
            errs.append(f"site {s} executed in {saw} but NOT in "
                        f"{[t for t in live if t not in saw]} -- the arms ran "
                        f"different code paths, so the delta is not attributable")

    want = {"base": "dense_gemm_tc", "base2": "dense_gemm_tc", "cublas": "cublaslt"}
    for t in arms:
        exp = want.get(t)
        r = rt[t] or {}
        if exp and r:
            bad = {s: v for s, v in r.items() if v != exp}
            if bad:
                errs.append(f"{t}: expected every site on '{exp}', got {bad} -- "
                            f"the env gate did not take effect")
    if not errs:
        print("    VERDICT: all three sites dispatched, and the gate flipped all three.")

    # ---------------------------------------------------------------- [2]
    print("\n[2] CONTAMINATION -- arrival census, scoped to the measurement window")
    cen = {t: census(f"{d}/serve-{t}.log") for t in arms}
    present = {k: v for k, v in cen.items() if v}
    if len(present) < 3:
        print("    fewer than 3 arms present -- no consensus available; "
              "contamination UNCHECKED (which is not the same as clean).")
    else:
        ours, foreign = split_foreign(present)
        for t in sorted(present):
            slope_reqs = [r for r in ours[t] if r[2] == SLOPE_MT]
            if not slope_reqs:
                errs.append(f"{t}: no max_tokens={SLOPE_MT} requests -- the slope "
                            f"sweep did not run")
                continue
            w0, w1 = slope_reqs[0][0], slope_reqs[-1][0]
            # fifo + max_batch=1: only foreign work INTERLEAVED into the window
            # can delay ours. Anything arriving after w1 queues behind us.
            inside = [r for r in foreign[t] if w0 <= r[0] <= w1]
            after = [r for r in foreign[t] if r[0] > w1]
            before = [r for r in foreign[t] if r[0] < w0]
            tag = "clean" if not (inside or before) else "CONTENDED"
            print(f"    {t:<8} {len(ours[t]):>3} ours / {len(foreign[t]):>2} foreign"
                  f"   window {len(slope_reqs)} reqs"
                  f"   in-window={len(inside)} pre={len(before)} post={len(after)}"
                  f"   {tag}")
            if inside or before:
                errs.append(
                    f"{t}: {len(inside)} foreign request(s) INSIDE the slope window"
                    f" (+{len(before)} already in flight at its start) -- this "
                    f"arm's slope is contended and void")
            elif after:
                print(f"             {len(after)} foreign arrival(s) AFTER the "
                      f"window: they queue behind us under fifo/max_batch=1, so "
                      f"the slope stands. This arm's decode TIMING is contended "
                      f"and must not be quoted. Its hashes are unaffected -- "
                      f"max_batch=1 precludes co-batching, so foreign traffic "
                      f"cannot reach our logits; contention moves clocks, not "
                      f"values.")
        # The sweep is deterministic. If the per-arm request shapes are not
        # identical, the arms were not asked the same question.
        shapes = {t: sorted((r[1], r[2]) for r in ours[t] if r[2] == SLOPE_MT)
                  for t in present}
        if len({tuple(v) for v in shapes.values()}) != 1:
            errs.append("the max_tokens=1 sweeps differ across arms -- the arms "
                        "were not given identical prompts, so the deltas are not "
                        "comparable")
        else:
            n = len(next(iter(shapes.values())))
            print(f"    all arms received an IDENTICAL {n}-request slope sweep")

    # ---------------------------------------------------------------- [4]
    print("\n[4] CORRECTNESS -- long prompts, through the patched GEMMs")
    hs = {t: load(f"{d}/hash-{t}.json") for t in arms}
    for t in arms:
        if hs[t] is None:
            errs.append(f"{t}: no hash artifact -- correctness UNGRADED (a missing "
                        f"row is not a passing row)")
            print(f"    {t:<8} NO HASH ARTIFACT -- ungraded")
    ref = hs.get("base")
    if ref:
        nb = sum(r["needle_found"] for r in ref["results"])
        print(f"    base needle recall {nb}/{len(ref['results'])}"
              + ("" if nb else "   <- oracle BLANK"))
        if not nb and "cublas" in arms:
            # For a bit-exact lever, base missing every needle is survivable:
            # hash equality still grades it. Here it is fatal. cuBLASLt reorders
            # accumulation, so cublas hashes are EXPECTED to differ and carry no
            # signal -- with the needle blank too there is no correctness
            # channel left, and "no channel" must not read as "no problem".
            errs.append("base recalled 0 needles, so the only oracle that can "
                        "grade a NON-bit-exact route is blank -- correctness is "
                        "UNGRADED, not passing. Raise --max-tokens on the probe "
                        "or shorten the filler, and re-run before quoting speed.")
        for t in arms:
            if t == "base" or not hs[t]:
                continue
            same = sum(x["hash"] == y["hash"]
                       for x, y in zip(hs[t]["results"], ref["results"]))
            nn = sum(r["needle_found"] for r in hs[t]["results"])
            tot = len(ref["results"])
            note = ""
            if t == "base2" and same < tot:
                note = ("   <- A/A HASHES MOVED: hashes are not stable arm-to-arm "
                        "here, so hash inequality carries NO information about cublas")
                errs.append("base2: A/A output hashes differ from base -- the hash "
                            "channel is not a usable discriminator in this session")
            elif t == "cublas":
                note = ("   (cuBLASLt reorders accumulation -- hash inequality is "
                        "EXPECTED and not a failure; the needle is the oracle)")
            print(f"    {t:<8} {same}/{tot} hashes match base, needle {nn}/{tot}{note}")
            if t == "cublas" and nb and nn < nb:
                errs.append(f"cublas: needle recall DROPPED {nb} -> {nn} -- the new "
                            f"route is losing information, not just reordering it")

    # ---------------------------------------------------------------- [3]
    print("\n[3] PREFILL THROUGHPUT (slope, per-request overhead removed)")
    sl = {}
    for t in arms:
        p = load(f"{d}/prefill-{t}.json")
        if not p:
            errs.append(f"{t}: no prefill artifact")
            continue
        s = slope(p)
        if not s:
            errs.append(f"{t}: prefill artifact carries no usable fit")
            continue
        sl[t] = s
        tok_s, r2, b, maxn = s
        flag = "" if r2 >= R2_MIN else f"   <- R^2 {r2:.4f} BELOW {R2_MIN}, unusable"
        if r2 < R2_MIN:
            errs.append(f"{t}: slope fit R^2={r2:.4f} < {R2_MIN}")
        print(f"    {t:<8} {tok_s:>7.0f} tok/s   (R^2={r2:.4f}, "
              f"intercept={b:.0f} ms, max N={maxn}){flag}")

    broken = bool(errs)
    if "base" in sl:
        base_v = sl["base"][0]
        floor = None
        if "base2" in sl:
            floor = abs(sl["base2"][0] - base_v) / base_v * 100.0
            print(f"\n    MEASURED A/A FLOOR (base vs base2): {floor:.1f}%"
                  f"   [prior assumption was {NOISE_PCT}%]")
        gate = max(floor, NOISE_PCT) if floor is not None else NOISE_PCT
        for t in arms:
            if t in ("base", "base2") or t not in sl:
                continue
            delta = (sl[t][0] - base_v) / base_v * 100.0
            if broken:
                print(f"    {t} vs base: {delta:+.1f}%   <- UNQUOTABLE, see the "
                      f"refusals below; this run did not earn a verdict")
            elif delta > gate:
                print(f"    {t} vs base: {delta:+.1f}%   REAL WIN "
                      f"(exceeds the {gate:.1f}% floor)")
            elif delta < -gate:
                print(f"    {t} vs base: {delta:+.1f}%   REAL LOSS")
            else:
                print(f"    {t} vs base: {delta:+.1f}%   within the {gate:.1f}% "
                      f"floor -- NULL, not a win")

    if errs:
        print("\n" + "=" * 74)
        print(f"REFUSING TO SCORE -- {len(errs)} problem(s):")
        for e in errs:
            print(f"  * {e}")
        print("=" * 74)
        return 1
    print("\nAll gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
