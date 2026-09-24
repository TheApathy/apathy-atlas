#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Verdict for tools/ctx_restore/gate.sh.

usage: compare.py <out_dir> <C:N>...
"""
import glob
import json
import os
import re
import struct
import sys


def logits(arm_dir, n):
    hits = sorted(glob.glob(os.path.join(arm_dir, f"*-{n}.bin")))
    return open(hits[-1], "rb").read() if hits else None


def text(arm_dir, name):
    try:
        r = json.load(open(os.path.join(arm_dir, name + ".resp.json")))
        return r["choices"][0]["text"], r.get("usage", {}).get("time_to_first_token_ms")
    except Exception as e:  # noqa: BLE001 - report, don't crash the verdict
        return f"<bad response: {e}>", None


def log(arm_dir):
    try:
        return open(os.path.join(arm_dir, "server.log"), errors="replace").read()
    except OSError:
        return ""


def bf16_stats(a, b):
    fa = [struct.unpack("<f", b"\0\0" + a[i:i + 2])[0] for i in range(0, len(a), 2)]
    fb = [struct.unpack("<f", b"\0\0" + b[i:i + 2])[0] for i in range(0, len(b), 2)]
    diff = max(abs(x - y) for x, y in zip(fa, fb))
    return diff, fa.index(max(fa)), fb.index(max(fb))


def main():
    out = sys.argv[1]
    cases = [tuple(int(x) for x in c.split(":")) for c in sys.argv[2:]]
    ok = True
    rows = []
    for c, n in cases:
        name = f"p2-{c}-{n}"
        r, f, k = (logits(os.path.join(out, a), n) for a in ("restore", "reference", "ctl_conv"))
        (rt, rttft), (ft, fttft) = text(os.path.join(out, "restore"), name), text(os.path.join(out, "reference"), name)
        same = r is not None and r == f
        tsame = rt == ft
        conv_differs = k is not None and f is not None and k != f
        ok &= same and tsame and conv_differs
        rows.append(f"case C={c} N={n}: logits restore==reference {same}; output text equal {tsame}; "
                    f"conv-omitted control differs {conv_differs}; TTFT restore {rttft} ms vs "
                    f"reference {fttft} ms")
    for arm, pat, want in [
        ("restore", r"ctx-cache: RESTORED", len(cases)),
        ("reference", r"REFERENCE mode", len(cases)),
        ("ctl_conv", r"GATE CONTROL", len(cases)),
        ("ctl_bad", r"REJECTED and deleted", 2),
    ]:
        got = len(re.findall(pat, log(os.path.join(out, arm))))
        good = got == want
        ok &= good
        rows.append(f"{arm}: '{pat}' x{got} (want {want}) {'ok' if good else 'FAIL'}")
    for arm in ("reference", "ctl_bad", "ctl_key"):
        if "ctx-cache: RESTORED" in log(os.path.join(out, arm)):
            ok = False
            rows.append(f"{arm}: restored a checkpoint it must not have: FAIL")
    kl = log(os.path.join(out, "ctl_key"))
    key_ok = "entries=0" in kl and "RESTORED" not in kl
    ok &= key_ok
    rows.append(f"ctl_key: wrong recipe sees zero checkpoints {key_ok}")
    for line in re.findall(r"ctx-cache: (?:RESTORED|captured|checkpoint written)[^\n]*", "\n".join(
            log(os.path.join(out, a)) for a in ("write", "restore"))):
        rows.append("  " + line)
    # Informational: how much does the extra boundary at C move the logits?
    # ctl_key ran the forged-case prompt as a plain full prefill.
    if len(cases) > 1:
        c, n = cases[1]
    else:
        c, n = cases[0]
    full, ref = logits(os.path.join(out, "ctl_key"), n), logits(os.path.join(out, "reference"), n)
    if full and ref and len(full) == len(ref):
        d, am_full, am_ref = bf16_stats(full, ref)
        rows.append(f"info: full prefill vs split-at-C (N={n}): bytes equal {full == ref}, "
                    f"max |dlogit| {d:.4g}, argmax {am_full} vs {am_ref}")
    rows.append("VERDICT: " + ("PASS" if ok else "FAIL"))
    print("\n".join(rows))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
