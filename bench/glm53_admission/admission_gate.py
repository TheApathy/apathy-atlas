#!/usr/bin/env python3
"""GLM-5.3 target-only admission gate: Atlas logits vs the ExLlamaV3 reference.

  admission_gate.py prefill <prompts.json> <ref_dir> <atlas_logits_dir> [--json out.json]
  admission_gate.py decode  <sequences.json> <ref_dir> <atlas_logits_dir> [--json out.json]
  admission_gate.py selftest

prefill: <prompts.json> is the list the reference ran on. The Atlas dump holds, in
  counter order, one or more all-rows chunks per prompt (logits-<n>-r<rows>.bf16);
  chunks are concatenated until they cover the prompt. Every row but the last is
  scored against the true next token.
decode: <sequences.json> is written by decode_sequences.py: per prompt, the prompt
  length and the full prompt+generated ids the reference ran on. Atlas rows are the
  prompt's last prefill row followed by one r1 dump per decode walk. There are no
  ground-truth tokens for a free-running continuation, so decode is judged on
  argmax agreement and KL only.

Bounds (pre-registered 2026-09-23; mirrored in target_only_admission.rs):
  argmax agreement excluding bf16 ties >= 0.90, mean KL(ref||atlas) <= 0.05,
  |top-1 delta| <= 0.01 absolute, |NLL delta| <= 2% relative, >= 3 prompts.
Exit 0 PASS, 1 FAIL, 2 UNAVAILABLE (missing/mismatched inputs are never a pass).
"""
import glob
import json
import os
import re
import sys

import numpy as np

V = 154_880
MIN_AGREE, MAX_KL, MAX_TOP1, MAX_NLL_REL, MIN_PROMPTS = 0.90, 0.05, 0.01, 0.02, 3


def unavailable(msg):
    print(f"UNAVAILABLE: {msg}")
    sys.exit(2)


def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_rne(x):
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def atlas_chunks(d):
    files = []
    for f in glob.glob(os.path.join(d, "logits-*.bf16")):
        m = re.fullmatch(r"logits-(\d+)(?:-r(\d+))?\.bf16", os.path.basename(f))
        if not m:
            unavailable(f"unrecognised dump name {f}")
        rows = int(m.group(2) or 1)
        if os.path.getsize(f) != rows * V * 2:
            unavailable(f"{f} is {os.path.getsize(f)} bytes, expected {rows * V * 2}")
        files.append((int(m.group(1)), rows, f))
    files.sort()
    if [n for n, _, _ in files] != list(range(len(files))):
        unavailable(f"dump counter is not contiguous in {d}")
    return files


def load_atlas(path, rows):
    return bf16_to_f32(np.fromfile(path, dtype=np.uint16)).reshape(rows, V)


def load_ref(d, i, rows):
    path = os.path.join(d, f"ref-{i}-r{rows}.f32")
    if not os.path.isfile(path) or os.path.getsize(path) != rows * V * 4:
        unavailable(f"reference {path} missing or wrong size")
    return np.fromfile(path, dtype=np.float32).reshape(rows, V)


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(1, keepdims=True))


def tied(x):
    top2 = np.partition(x, -2, axis=1)[:, -2:]
    return top2[:, 0] == top2[:, 1]


def score(ref, atl, truth):
    """Per-row stats for one prompt. truth is None for free-running decode."""
    rlp, alp = log_softmax(ref), log_softmax(atl)
    ties = tied(atl) | tied(f32_to_bf16_rne(ref))
    live = ~ties
    agree = (ref.argmax(1) == atl.argmax(1))[live]
    kl = (np.exp(rlp) * (rlp - alp)).sum(1)
    out = {"rows": int(len(ref)), "tied_rows": int(ties.sum()),
           "agree_sum": float(agree.sum()), "agree_n": int(live.sum()), "kl_sum": float(kl.sum()),
           "kl_max": float(kl.max())}
    if truth is not None:
        idx = np.arange(len(truth))
        out.update({
            "top1_ref": float((ref.argmax(1) == truth).mean()),
            "top1_atlas": float((atl.argmax(1) == truth).mean()),
            "nll_ref": float(-rlp[idx, truth].mean()),
            "nll_atlas": float(-alp[idx, truth].mean()),
        })
    return out


def aggregate(per, with_truth):
    agree_n = sum(p["agree_n"] for p in per)
    rows = sum(p["rows"] for p in per)
    m = {
        "prompts": len(per),
        "positions": rows,
        "argmax_agreement_excl_ties": sum(p["agree_sum"] for p in per) / max(agree_n, 1),
        "mean_kl": sum(p["kl_sum"] for p in per) / max(rows, 1),
        "max_kl": max(p["kl_max"] for p in per),
        "tied_rows": sum(p["tied_rows"] for p in per),
    }
    if with_truth:
        t1r = np.mean([p["top1_ref"] for p in per])
        t1a = np.mean([p["top1_atlas"] for p in per])
        nr = np.mean([p["nll_ref"] for p in per])
        na = np.mean([p["nll_atlas"] for p in per])
        m.update({"top1_ref": float(t1r), "top1_atlas": float(t1a),
                  "top1_delta_abs": float(abs(t1a - t1r)),
                  "nll_ref": float(nr), "nll_atlas": float(na),
                  "nll_delta_rel": float(abs(na - nr) / nr)})
    return m


def verdict(m):
    fails = []
    if m["prompts"] < MIN_PROMPTS:
        fails.append(f"prompts {m['prompts']} < {MIN_PROMPTS}")
    if m["argmax_agreement_excl_ties"] < MIN_AGREE:
        fails.append(f"agreement {m['argmax_agreement_excl_ties']:.4f} < {MIN_AGREE}")
    if m["mean_kl"] > MAX_KL:
        fails.append(f"mean KL {m['mean_kl']:.5f} > {MAX_KL}")
    if "top1_delta_abs" in m and m["top1_delta_abs"] > MAX_TOP1:
        fails.append(f"top-1 delta {m['top1_delta_abs']:.4f} > {MAX_TOP1}")
    if "nll_delta_rel" in m and m["nll_delta_rel"] > MAX_NLL_REL:
        fails.append(f"NLL delta {m['nll_delta_rel']:.4f} > {MAX_NLL_REL}")
    return fails


def run_prefill(prompts_path, ref_dir, atlas_dir):
    prompts = json.load(open(prompts_path))
    chunks = atlas_chunks(atlas_dir)
    per, c = [], 0
    for i, p in enumerate(prompts):
        n = len(p["ids"])
        parts, have = [], 0
        while have < n:
            if c >= len(chunks):
                unavailable(f"Atlas dump ran out at prompt {i}")
            _, rows, f = chunks[c]
            parts.append(load_atlas(f, rows)); have += rows; c += 1
        if have != n:
            unavailable(f"prompt {i}: Atlas chunks cover {have} rows, prompt has {n}")
        atl = np.concatenate(parts)
        ref = load_ref(ref_dir, i, n)
        truth = np.array(p["ids"][1:])
        s = score(ref[:-1], atl[:-1], truth)
        s["name"] = p["name"]
        per.append(s)
    if c != len(chunks):
        unavailable(f"{len(chunks) - c} unexplained Atlas dumps after the prompts")
    return per, True


def run_decode(seq_path, ref_dir, atlas_dir):
    seqs = json.load(open(seq_path))
    per = []
    for i, s in enumerate(seqs):
        n, plen = len(s["ids"]), s["prompt_len"]
        ref = load_ref(ref_dir, i, n)
        atl = np.stack([load_atlas(f, 1)[0] if r == 1 else load_atlas(f, r)[-1]
                        for f, r in zip(s["atlas_files"], s["atlas_rows"])])
        # Atlas row k predicts generated token k; the reference predicts it at
        # position plen - 1 + k. The final generated token is never fed back.
        k = len(atl)
        if plen - 1 + k > n:
            unavailable(f"sequence {i}: {k} Atlas rows exceed the reference")
        st = score(ref[plen - 1: plen - 1 + k], atl, None)
        st["name"] = s["name"]
        per.append(st)
    return per, False


def selftest():
    rng = np.random.default_rng(0)
    # Model-like rows: one clear winner per row. Pure noise over 154,880 logits
    # has top-2 margins below bf16 resolution, which measures the rounding, not
    # the gate.
    ref = rng.normal(0, 3, (64, V)).astype(np.float32)
    truth = rng.integers(0, V, 64)
    ref[np.arange(64), truth] += 20.0
    same = score(ref, f32_to_bf16_rne(ref), truth)
    m = aggregate([same] * 3, True)
    assert not verdict(m), f"bf16 copy of the reference must pass: {m}"
    wrong = score(ref, rng.normal(0, 3, (64, V)).astype(np.float32), truth)
    m = aggregate([wrong] * 3, True)
    assert verdict(m), f"unrelated logits must fail: {m}"
    assert aggregate([same] * 2, True)["prompts"] == 2 and verdict(aggregate([same] * 2, True))
    print("selftest ok: identical passes, unrelated fails, 2 prompts refused")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "selftest":
        return selftest()
    if len(sys.argv) < 5:
        unavailable(__doc__)
    mode, spec, ref_dir, atlas_dir = sys.argv[1:5]
    out = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
    if mode == "prefill":
        per, with_truth = run_prefill(spec, ref_dir, atlas_dir)
    elif mode == "decode":
        per, with_truth = run_decode(spec, ref_dir, atlas_dir)
    else:
        unavailable(f"unknown mode {mode}")
    m = aggregate(per, with_truth)
    fails = verdict(m)
    for p in per:
        a = p["agree_sum"] / max(p["agree_n"], 1)
        print(f"  {p['name']:>14}: rows={p['rows']} agree={a:.4f} ties={p['tied_rows']} "
              f"meanKL={p['kl_sum'] / p['rows']:.5f} maxKL={p['kl_max']:.3f}"
              + (f" top1 ref/atl={p['top1_ref']:.4f}/{p['top1_atlas']:.4f}"
                 f" nll ref/atl={p['nll_ref']:.4f}/{p['nll_atlas']:.4f}" if with_truth else ""))
    print(json.dumps(m, indent=2))
    print("PASS" if not fails else "FAIL: " + "; ".join(fails))
    if out:
        json.dump({"mode": mode, "metrics": m, "per_prompt": per, "fails": fails}, open(out, "w"), indent=2)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
